//! MMTk VM binding for the julia-mlir compiler's object model.
//!
//! This module provides the VMBinding trait implementation for binaries
//! compiled by julia-mlir. The object model differs from stock Julia:
//! type descriptors with nfields/instance_size replace jl_datatype_t,
//! and a shadow stack replaces Julia's GC frame mechanism.
//!
//! Architecture:
//! - Single-threaded initially (single GC worker for Immix/LXR).
//! - Object model: [tag_word 8B | body NB]. Tag = raw DataType pointer.
//!   MMTk side metadata handles GC bits; the tag word has no GC metadata.
//! - No dependency on Julia's C runtime.
//!
//! Plan ladder (selected via Cargo features):
//!   julia_mlir_nogc          → NoGC (bring-up, allocation only)
//!   julia_mlir_immix         → Non-moving Immix (first real GC)
//!   julia_mlir_immix_moving  → Moving Immix (objects relocate)
//!   julia_mlir_lxr           → LXR (RC + concurrent tracing, destination plan)
//!
//! Object layout:
//!   [tag_word 8B | field_0 8B | field_1 8B | ... | field_N 8B]
//!
//!   - tag_word: pointer to the object's type descriptor (DataType)
//!   - The type descriptor contains nfields and field layout info for scanning
//!   - ObjectReference points to the start of field_0 (body start, tag at -8)
//!
//! Type descriptor layout (80 bytes, matches Standalone.jl TYPEDESC_*):
//!   [0]  name_ptr        i64  -> Symbol for the type name
//!   [8]  instance_size   i64  -> Size of body in bytes (0 for abstract)
//!   [16] nfields         i64  -> Number of pointer-sized fields
//!   [24] super_ptr       i64  -> Pointer to supertype descriptor
//!   [32] flags           i64  -> Bitfield (abstract, mutable, has_params)
//!   [40] hash            i64  -> Type hash
//!   [48] field_types_ptr i64  -> Pointer to array of type descriptors per field
//!   [56] field_offsets_ptr i64 -> Pointer to array of field byte offsets
//!   [64] params_ptr      i64  -> Type parameters (or null)
//!   [72] uid             i64  -> Unique type ID

use mmtk::scheduler::RootKind;
use mmtk::util::opaque_pointer::*;
use mmtk::util::{Address, ObjectReference};
use mmtk::vm::slot::SimpleSlot;
use mmtk::vm::*;
use mmtk::AllocationSemantics;
use mmtk::{memory_manager, MMTKBuilder, Mutator, MMTK};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ============================================================================
// Type descriptor layout constants (must match Standalone.jl)
// ============================================================================

const TYPEDESC_OFF_INSTANCE_SZ: usize = 8;
const TYPEDESC_OFF_NFIELDS: usize = 16;
const TYPEDESC_OFF_FLAGS: usize = 32;
/// Pointer bitmap: bit i set ⇔ word i of the body is a managed pointer.
/// Supports up to 64 fields. Zero means no pointer fields.
const TYPEDESC_OFF_PTR_BITMAP: usize = 48;
/// Element size (for array types): size in bytes of each element.
/// Zero means not an array type.
const TYPEDESC_OFF_ELEM_SIZE: usize = 56;

const TYPE_FLAG_ABSTRACT: u64 = 0x1;
/// Flag indicating this type is a variable-size array (data stored inline).
/// Array body layout: [length 8B | capacity 8B | elem_type_ptr 8B | data...]
const TYPE_FLAG_VARSIZE: u64 = 0x8;
/// Flag indicating this type is a Memory (fixed-size inline data buffer).
/// Memory body layout: [length 8B | data[0..length*elem_size]]
/// Simpler than VARSIZE arrays: no capacity, no elem_type_ptr.
const TYPE_FLAG_MEMORY: u64 = 0x10;

// ============================================================================
// VM binding type
// ============================================================================

/// The standalone Julia VM binding — no C runtime, no jl_tls, no tasks.
#[derive(Default)]
pub struct JuliaMlirVM;

impl VMBinding for JuliaMlirVM {
    /// Julia objects are 16-byte aligned (matching jl_gc_alloc).
    const MAX_ALIGNMENT: usize = 16;
    const MIN_ALIGNMENT: usize = 8;

    type VMObjectModel = JuliaMlirObjectModel;
    type VMScanning = JuliaMlirScanning;
    type VMCollection = JuliaMlirCollection;
    type VMActivePlan = JuliaMlirActivePlan;
    type VMReferenceGlue = JuliaMlirReferenceGlue;
    type VMMemorySlice = mmtk::vm::slot::UnimplementedMemorySlice<SimpleSlot>;
    type VMSlot = SimpleSlot;
}

// ============================================================================
// Object model
// ============================================================================

/// Object layout: [tag_word 8B | body NB]
///
/// ObjectReference points to the start of the body (offset +8 from allocation
/// start). The tag word at offset -8 holds a raw DataType pointer.
///
/// GC metadata (mark bits, forwarding, pinning) is stored in MMTk's side
/// metadata tables, indexed by object address. The tag word is clean —
/// no GC bits encoded in it.
pub struct JuliaMlirObjectModel;

// Side metadata specs — must match the ordering expected by mmtk-core.
pub(crate) const LOGGING_SIDE_METADATA_SPEC: VMGlobalLogBitSpec = VMGlobalLogBitSpec::side_first();

pub(crate) const FIELD_UNLOG_SIDE_METADATA_SPEC: VMGlobalFieldUnlogBitSpec =
    VMGlobalFieldUnlogBitSpec::side_after(LOGGING_SIDE_METADATA_SPEC.as_spec());

pub(crate) const LOS_METADATA_SPEC: VMLocalLOSMarkNurserySpec =
    VMLocalLOSMarkNurserySpec::side_first();

pub(crate) const MARKING_METADATA_SPEC: VMLocalMarkBitSpec =
    VMLocalMarkBitSpec::side_after(LOS_METADATA_SPEC.as_spec());

impl ObjectModel<JuliaMlirVM> for JuliaMlirObjectModel {
    const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec = LOGGING_SIDE_METADATA_SPEC;
    const GLOBAL_FIELD_UNLOG_BIT_SPEC: VMGlobalFieldUnlogBitSpec = FIELD_UNLOG_SIDE_METADATA_SPEC;

    const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec =
        VMLocalForwardingPointerSpec::in_header(-64);

    // Forwarding bits in the header: use the lowest 2 bits of the tag word
    // (same location as the forwarding pointer, offset -64 bits from ObjectRef).
    // The tag word holds a DataType pointer with 16-byte alignment, so the
    // bottom 4 bits are available. LXR's RC-enabled Immix space does NOT
    // include forwarding bits in its local side-metadata list, so putting
    // them in side metadata would leave them unmapped → SIGSEGV.
    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec =
        VMLocalForwardingBitsSpec::in_header(-64);

    const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec = MARKING_METADATA_SPEC;
    const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec = LOS_METADATA_SPEC;

    const UNIFIED_OBJECT_REFERENCE_ADDRESS: bool = false;
    const OBJECT_REF_OFFSET_LOWER_BOUND: isize = 0;

    fn copy(
        from: ObjectReference,
        semantics: mmtk::util::copy::CopySemantics,
        copy_context: &mut mmtk::util::copy::GCWorkerCopyContext<JuliaMlirVM>,
    ) -> ObjectReference {
        let bytes = Self::get_current_size(from);
        let from_start = Self::ref_to_object_start(from);
        let header_offset = 8usize; // tag word is always 8 bytes

        let dst = copy_context.alloc_copy(from, bytes, 16, 8, semantics);
        debug_assert!(!dst.is_zero());

        unsafe {
            std::ptr::copy_nonoverlapping::<u8>(from_start.to_ptr(), dst.to_mut_ptr(), bytes);
        }

        let to_obj = unsafe { ObjectReference::from_raw_address_unchecked(dst + header_offset) };
        copy_context.post_copy(to_obj, bytes, semantics);
        to_obj
    }

    fn copy_to(_from: ObjectReference, _to: ObjectReference, _region: Address) -> Address {
        unimplemented!("copy_to not needed for Immix/LXR")
    }

    fn get_current_size(object: ObjectReference) -> usize {
        // Read size from the type descriptor.
        // tag_word at (obj - 8), type descriptor at *tag_word.
        //
        // For fixed-size objects: total = 8 + instance_size
        // For variable-size (array) objects (TYPE_FLAG_VARSIZE set):
        //   Body layout: [length 8B | capacity 8B | elem_type_ptr 8B | data[0..capacity*elem_size]]
        //   total = 8 + 24 + capacity * elem_size
        unsafe {
            let tag_addr = object.to_raw_address() - 8usize;
            let type_desc = Address::from_usize(tag_addr.load::<usize>());
            if type_desc.is_zero() {
                // Defensive: no type descriptor means we can't determine size.
                // Return minimum object size (tag + 16-byte align).
                return 16;
            }

            let flags = (type_desc + TYPEDESC_OFF_FLAGS).load::<u64>();
            if flags & TYPE_FLAG_MEMORY != 0 {
                // Memory object (fixed-size inline data buffer).
                // Body: [length 8B | data[0..length*elem_size]]
                let obj_addr = object.to_raw_address();
                let length = obj_addr.load::<u64>() as usize; // offset 0 in body
                let elem_size = (type_desc + TYPEDESC_OFF_ELEM_SIZE).load::<u64>() as usize;
                let header_size = 8; // length only
                let data_size = length * elem_size;
                let total = 8 + header_size + data_size; // tag + header + data
                (total + 15) & !15
            } else if flags & TYPE_FLAG_VARSIZE != 0 {
                // Variable-size (array) object.
                // Body: [length 8B | capacity 8B | elem_type_ptr 8B | data...]
                let obj_addr = object.to_raw_address();
                let capacity = (obj_addr + 8usize).load::<u64>() as usize; // offset 8 in body
                let elem_size = (type_desc + TYPEDESC_OFF_ELEM_SIZE).load::<u64>() as usize;
                let header_size = 24; // length + capacity + elem_type_ptr
                let data_size = capacity * elem_size;
                let total = 8 + header_size + data_size; // tag + header + data
                (total + 15) & !15
            } else {
                let instance_size = (type_desc + TYPEDESC_OFF_INSTANCE_SZ).load::<u64>() as usize;
                // Total allocation size: tag_word (8B) + body, aligned to 16
                let total = 8 + instance_size;
                (total + 15) & !15
            }
        }
    }

    fn get_size_when_copied(object: ObjectReference) -> usize {
        Self::get_current_size(object)
    }

    fn get_align_when_copied(_object: ObjectReference) -> usize {
        16
    }

    fn get_align_offset_when_copied(_object: ObjectReference) -> usize {
        0
    }

    fn get_reference_when_copied_to(_from: ObjectReference, to: Address) -> ObjectReference {
        unsafe { ObjectReference::from_raw_address_unchecked(to + 8usize) }
    }

    fn get_type_descriptor(_reference: ObjectReference) -> &'static [i8] {
        unimplemented!()
    }

    #[inline(always)]
    fn ref_to_object_start(object: ObjectReference) -> Address {
        // Allocation start: [tag 8B | body]
        // obj points to body, so start is obj - 8.
        object.to_raw_address() - 8usize
    }

    #[inline(always)]
    fn ref_to_header(object: ObjectReference) -> Address {
        object.to_raw_address()
    }

    fn dump_object(_object: ObjectReference) {}

    fn dump_object_s(_object: ObjectReference) -> String {
        String::from("<standalone object>")
    }

    fn get_class_pointer(object: ObjectReference) -> Address {
        // Tag word is at (obj - 8)
        unsafe {
            let tag_addr = object.to_raw_address() - 8usize;
            Address::from_usize(tag_addr.load::<usize>())
        }
    }
}

// ============================================================================
// Scanning — real object scanning for Immix/LXR
// ============================================================================

/// Classify an object as Scalar, ObjArray, or ValArray by reading its type
/// descriptor once.  All three trait methods (`is_obj_array`, `is_val_array`,
/// `get_obj_kind`) delegate here so we never redundantly reload the tag.
#[inline(always)]
fn classify_object(o: ObjectReference) -> ObjectKind {
    unsafe {
        let tag_addr = o.to_raw_address() - 8usize;
        let type_desc = Address::from_usize(tag_addr.load::<usize>());
        if type_desc.is_zero() {
            return ObjectKind::Scalar;
        }
        let flags = (type_desc + TYPEDESC_OFF_FLAGS).load::<u64>();

        // Memory{T}: fixed-size inline data, classified like arrays for GC.
        if flags & TYPE_FLAG_MEMORY != 0 {
            let ptr_bitmap = (type_desc + TYPEDESC_OFF_PTR_BITMAP).load::<u64>();
            if ptr_bitmap & 1 != 0 {
                let elem_size = (type_desc + TYPEDESC_OFF_ELEM_SIZE).load::<u64>() as u32;
                ObjectKind::ObjArray(elem_size)
            } else {
                ObjectKind::ValArray
            }
        } else if flags & TYPE_FLAG_VARSIZE != 0 {
            let ptr_bitmap = (type_desc + TYPEDESC_OFF_PTR_BITMAP).load::<u64>();
            if ptr_bitmap & 1 != 0 {
                let elem_size = (type_desc + TYPEDESC_OFF_ELEM_SIZE).load::<u64>() as u32;
                ObjectKind::ObjArray(elem_size)
            } else {
                ObjectKind::ValArray
            }
        } else {
            ObjectKind::Scalar
        }
    }
}

pub struct JuliaMlirScanning;

impl Scanning<JuliaMlirVM> for JuliaMlirScanning {
    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        _mutator: &'static mut Mutator<JuliaMlirVM>,
        mut factory: impl RootsWorkFactory<SimpleSlot>,
    ) {
        // Enumerate roots from the GC root stack.
        // The MLIR-compiled code maintains a shadow stack at a well-known global:
        //   GC_ROOT_STACK_TOP_ADDR -> address of the @_jlmlir_gc_root_top global
        //                             (dereference to get current top value)
        //   GC_ROOT_STACK_BASE    -> base address of the root stack array
        // Each entry on the stack is a pointer to a heap object (8 bytes).
        //
        // The MLIR code writes root_top directly to @_jlmlir_gc_root_top
        // without any FFI call. We read it through indirection here.
        unsafe {
            let top_addr = GC_ROOT_STACK_TOP_ADDR.load(Ordering::Relaxed);
            let base_ptr = GC_ROOT_STACK_BASE.load(Ordering::Relaxed);
            if top_addr == 0 || base_ptr == 0 {
                return;
            }
            // Dereference the top address to get the current top value
            let top_ptr = *(top_addr as *const usize);
            if top_ptr <= base_ptr {
                return;
            }

            let mut roots = Vec::new();
            let mut cur = base_ptr;
            while cur < top_ptr {
                let slot_addr = Address::from_usize(cur);
                let obj_addr = slot_addr.load::<usize>();
                if obj_addr != 0 {
                    if let Some(objref) =
                        ObjectReference::from_raw_address(Address::from_usize(obj_addr))
                    {
                        roots.push(objref);
                    }
                }
                cur += 8;
            }

            const CHUNK: usize = 4096;
            for chunk in roots.chunks(CHUNK) {
                factory.create_process_pinning_roots_work(chunk.to_vec());
            }
        }
    }

    fn scan_vm_specific_roots(
        _tls: VMWorkerThread,
        mut factory: impl RootsWorkFactory<SimpleSlot>,
    ) {
        // Scan the global root table — type descriptors, interned symbols,
        // global variables. These are registered via jlmlir_add_global_root().
        unsafe {
            let count = GLOBAL_ROOTS_COUNT.load(Ordering::Relaxed);
            if count == 0 {
                return;
            }

            let roots_base = GLOBAL_ROOTS_BASE.load(Ordering::Relaxed);
            if roots_base == 0 {
                return;
            }

            let mut roots = Vec::with_capacity(count);
            for i in 0..count {
                let slot_addr = Address::from_usize(roots_base + i * 8);
                let obj_addr = slot_addr.load::<usize>();
                if obj_addr != 0 {
                    if let Some(objref) =
                        ObjectReference::from_raw_address(Address::from_usize(obj_addr))
                    {
                        roots.push(objref);
                    }
                }
            }

            const CHUNK: usize = 4096;
            for chunk in roots.chunks(CHUNK) {
                factory.create_process_pinning_roots_work(chunk.to_vec());
            }
        }
    }

    fn scan_object(
        _tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut impl SlotVisitor<SimpleSlot>,
    ) {
        // Precise object scanning using the pointer bitmap in the type descriptor.
        //
        // Descriptor slot 6 (offset 48) holds a bitmap: bit i set ⇔ word i of
        // the body is a managed pointer.  Only those fields are reported to the
        // GC.  This is critical for LXR, whose RC pipeline (ProcessIncs) accesses
        // side metadata keyed by the target address — passing non-pointer data
        // (small integers, floats) as ObjectReferences causes unmapped-metadata
        // SIGSEGVs.
        //
        // If the bitmap is zero and nfields > 0, no fields are pointers (all
        // data).  If both nfields and bitmap are zero, there is nothing to scan.
        //
        // Variable-size (array) objects (TYPE_FLAG_VARSIZE set):
        //   Body: [length 8B | capacity 8B | elem_type_ptr 8B | data...]
        //   The header's elem_type_ptr (offset 16) is always a managed pointer.
        //   If the element type has pointer elements (checked via elem_size == 8
        //   and elem_type has ptr_bitmap or is boxed), scan all `length` data slots.
        //   For now: if elem_size == 8 (pointer-sized), scan all data slots.
        //   Unboxed element arrays (Int64, Float64) have elem_size == 8 but
        //   ptr_bitmap indicates whether elements are pointers.
        unsafe {
            let obj_addr = object.to_raw_address();
            let tag_addr = obj_addr - 8usize;
            let type_desc_addr = Address::from_usize(tag_addr.load::<usize>());
            if type_desc_addr.is_zero() {
                return;
            }

            let flags = (type_desc_addr + TYPEDESC_OFF_FLAGS).load::<u64>();
            if flags & TYPE_FLAG_ABSTRACT != 0 {
                return;
            }

            // Check for Memory objects (TYPE_FLAG_MEMORY).
            // Memory body: [length 8B | data[0..length*elem_size]]
            // No capacity, no elem_type_ptr — simpler than arrays.
            if flags & TYPE_FLAG_MEMORY != 0 {
                let ptr_bitmap = (type_desc_addr + TYPEDESC_OFF_PTR_BITMAP).load::<u64>();
                if ptr_bitmap & 1 != 0 {
                    // Elements are managed pointers — scan all `length` data slots.
                    let length = obj_addr.load::<u64>() as usize;
                    let data_start = obj_addr + 8usize; // after length header (1 × 8B)
                    for i in 0..length {
                        let slot_addr = data_start + (i * 8);
                        let slot = SimpleSlot::from_address(slot_addr);
                        slot_visitor.visit_slot(slot, false);
                    }
                }
                // If ptr_bitmap bit 0 is not set, elements are unboxed data — no scanning.
                return;
            }

            // Check for variable-size array objects (TYPE_FLAG_VARSIZE).
            if flags & TYPE_FLAG_VARSIZE != 0 {
                // Array body: [length 8B | capacity 8B | elem_type_ptr 8B | data...]
                // The elem_type_ptr at body offset 16 is always a managed pointer → scan it.
                let elem_type_slot_addr = obj_addr + 16usize;
                let elem_type_slot = SimpleSlot::from_address(elem_type_slot_addr);
                slot_visitor.visit_slot(elem_type_slot, false);

                // Check ptr_bitmap: bit 0 set ⇒ array elements are managed pointers.
                // This is a per-array-type flag (e.g., Vector{Any} has it, Vector{Int64} doesn't).
                let ptr_bitmap = (type_desc_addr + TYPEDESC_OFF_PTR_BITMAP).load::<u64>();
                if ptr_bitmap & 1 != 0 {
                    // Elements are managed pointers — scan all `length` data slots.
                    let length = obj_addr.load::<u64>() as usize;
                    let data_start = obj_addr + 24usize; // after header (3 × 8B)
                    for i in 0..length {
                        let slot_addr = data_start + (i * 8);
                        let slot = SimpleSlot::from_address(slot_addr);
                        slot_visitor.visit_slot(slot, false);
                    }
                }
                // If ptr_bitmap bit 0 is not set, elements are unboxed data — no scanning.
                return;
            }

            let ptr_bitmap = (type_desc_addr + TYPEDESC_OFF_PTR_BITMAP).load::<u64>();
            if ptr_bitmap == 0 {
                // No pointer fields — nothing to scan.
                return;
            }

            // Scan only the fields marked as pointers in the bitmap.
            let nfields = (type_desc_addr + TYPEDESC_OFF_NFIELDS).load::<u64>() as usize;
            let max_field = nfields.min(64);
            let mut bits = ptr_bitmap;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                if i >= max_field {
                    break;
                }
                let field_addr = obj_addr + (i * 8);
                let slot = SimpleSlot::from_address(field_addr);
                slot_visitor.visit_slot(slot, false);
                bits &= bits - 1; // clear lowest set bit
            }
        }
    }

    fn scan_object_with_klass(
        tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut impl SlotVisitor<SimpleSlot>,
        _klass: Address,
    ) {
        // Delegate to scan_object — klass is the type descriptor we already read from the tag.
        Self::scan_object(tls, object, slot_visitor);
    }

    fn scan_multiple_thread_root(
        _tls: VMWorkerThread,
        _mutators: Vec<VMMutatorThread>,
        mut factory: impl RootsWorkFactory<SimpleSlot>,
    ) {
        // Single-threaded: enumerate all roots from the shadow stack.
        // The root stack stores raw pointers to heap objects.
        unsafe {
            let top_addr = GC_ROOT_STACK_TOP_ADDR.load(Ordering::SeqCst);
            let base = GC_ROOT_STACK_BASE.load(Ordering::SeqCst);
            if top_addr == 0 || base == 0 {
                return;
            }
            let top = *(top_addr as *const usize);
            if top <= base {
                return;
            }

            let n_roots = (top - base) / 8;
            let mut roots = Vec::with_capacity(n_roots);
            let mut cur = base;
            while cur < top {
                let slot = SimpleSlot::from_address(Address::from_usize(cur));
                roots.push(slot);
                cur += 8;
            }

            const CHUNK: usize = 4096;
            for chunk in roots.chunks(CHUNK) {
                factory.create_process_roots_work(chunk.to_vec(), RootKind::Strong);
            }
        }
    }

    fn notify_initial_thread_scan_complete(_partial_scan: bool, _tls: VMWorkerThread) {}

    fn supports_return_barrier() -> bool {
        false
    }

    fn prepare_for_roots_re_scanning() {}

    fn is_obj_array(o: ObjectReference) -> bool {
        matches!(classify_object(o), ObjectKind::ObjArray(_))
    }

    fn is_val_array(o: ObjectReference) -> bool {
        matches!(classify_object(o), ObjectKind::ValArray)
    }

    fn get_obj_kind(o: ObjectReference) -> ObjectKind {
        classify_object(o)
    }

    fn process_weak_refs(
        _worker: &mut mmtk::scheduler::GCWorker<JuliaMlirVM>,
        _tracer_context: impl ObjectTracerContext<JuliaMlirVM>,
    ) -> bool {
        // No weak references in standalone (yet).
        false
    }
}

// ============================================================================
// Collection — real GC for Immix
// ============================================================================

pub struct JuliaMlirCollection;

/// Global flag: set by GC worker to request mutator stop.
static GC_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global flag: set by mutator when it has stopped at a safepoint.
static MUTATOR_STOPPED: AtomicBool = AtomicBool::new(false);

/// Global flag: set by GC worker to resume the mutator.
static GC_DONE: AtomicBool = AtomicBool::new(false);

impl Collection<JuliaMlirVM> for JuliaMlirCollection {
    fn stop_all_mutators<F>(
        _tls: VMWorkerThread,
        mut mutator_visitor: F,
        _current_gc_should_unload_classes: bool,
    ) where
        F: FnMut(&'static mut Mutator<JuliaMlirVM>),
    {
        // Report GC start (initializes the GC_START_TIME timer that
        // dump_gc_stats reads — the lxr-julia-v2 branch of mmtk-core
        // doesn't call report_gc_start automatically).
        memory_manager::report_gc_start(get_mmtk());

        // Signal the mutator to stop at the next safepoint.
        GC_REQUESTED.store(true, Ordering::SeqCst);

        // Arm the page-protection safepoint: mprotect to PROT_NONE so the
        // next volatile load from the safepoint page faults and triggers
        // the SIGSEGV handler which calls block_for_gc.
        if SAFEPOINT_PAGE.load(Ordering::Relaxed) != 0 {
            unsafe {
                jlmlir_safepoint_arm();
            }
        }

        // Wait for the mutator to reach a safepoint and stop.
        while !MUTATOR_STOPPED.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }

        // Visit the single mutator.
        unsafe {
            let mutator_ptr = THE_MUTATOR.load(Ordering::SeqCst);
            if mutator_ptr != 0 {
                let mutator = &mut *(mutator_ptr as *mut Mutator<JuliaMlirVM>);
                mutator_visitor(mutator);
            }
        }
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        GC_REQUESTED.store(false, Ordering::SeqCst);
        MUTATOR_STOPPED.store(false, Ordering::SeqCst);

        // Disarm the safepoint page: mprotect back to PROT_READ so future
        // volatile loads succeed without faulting.
        if SAFEPOINT_PAGE.load(Ordering::Relaxed) != 0 {
            unsafe {
                jlmlir_safepoint_disarm();
            }
        }

        GC_DONE.store(true, Ordering::SeqCst);
    }

    fn block_for_gc(_tls: VMMutatorThread) {
        // Called by the mutator when allocation fails and GC is needed.
        // Also called from explicit safepoint checks.
        // Signal that we've stopped, then wait for GC to complete.
        MUTATOR_STOPPED.store(true, Ordering::SeqCst);

        // Spin until GC is done.
        while !GC_DONE.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        GC_DONE.store(false, Ordering::SeqCst);

        // Track GC count for the gate test.
        GC_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    fn spawn_gc_thread(_tls: VMThread, ctx: GCThreadContext<JuliaMlirVM>) {
        // Spawn a real GC worker thread for Immix collection.
        let _ = std::thread::Builder::new()
            .name("MMTk Worker".to_string())
            .spawn(move || {
                let worker_tls = VMWorkerThread(VMThread(OpaquePointer::from_address(unsafe {
                    Address::from_usize(1)
                })));
                match ctx {
                    GCThreadContext::Worker(w) => {
                        mmtk::memory_manager::start_worker(get_mmtk(), worker_tls, w)
                    }
                }
            });
    }

    fn schedule_finalization(_tls: VMWorkerThread) {}

    fn out_of_memory(_tls: VMThread, _err_kind: mmtk::util::alloc::AllocationError) {
        // Write "FATAL: out of memory\n" via raw syscall, then abort.
        unsafe {
            let msg = b"FATAL: out of memory\n";
            libc::write(2, msg.as_ptr() as *const _, msg.len());
            libc::abort();
        }
    }

    fn vm_live_bytes() -> usize {
        0
    }

    fn is_collection_enabled() -> bool {
        // Always enabled for Immix — the whole point is to collect.
        true
    }
}

// ============================================================================
// Active plan — single-threaded, one mutator
// ============================================================================

pub struct JuliaMlirActivePlan;

/// Pointer to the single mutator (set by jlmlir_bind_mutator).
static THE_MUTATOR: AtomicUsize = AtomicUsize::new(0);

impl ActivePlan<JuliaMlirVM> for JuliaMlirActivePlan {
    fn number_of_mutators() -> usize {
        1
    }

    fn is_mutator(_tls: VMThread) -> bool {
        true
    }

    fn mutator(_tls: VMMutatorThread) -> &'static mut Mutator<JuliaMlirVM> {
        unsafe {
            let ptr = THE_MUTATOR.load(Ordering::Relaxed);
            &mut *(ptr as *mut Mutator<JuliaMlirVM>)
        }
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut Mutator<JuliaMlirVM>> + 'a> {
        let ptr = THE_MUTATOR.load(Ordering::Relaxed);
        if ptr == 0 {
            Box::new(std::iter::empty())
        } else {
            Box::new(std::iter::once(unsafe {
                &mut *(ptr as *mut Mutator<JuliaMlirVM>)
            }))
        }
    }

    fn vm_trace_object<Q: mmtk::plan::ObjectQueue>(
        queue: &mut Q,
        object: ObjectReference,
        _worker: &mut mmtk::scheduler::GCWorker<JuliaMlirVM>,
    ) -> ObjectReference {
        queue.enqueue(object);
        object
    }
}

// ============================================================================
// Reference glue — no weak/soft/phantom references in standalone
// ============================================================================

pub struct JuliaMlirReferenceGlue;

#[derive(Clone, Copy, Debug)]
pub struct DummyFinalizable(pub ObjectReference);

impl mmtk::vm::Finalizable for DummyFinalizable {
    fn get_reference(&self) -> ObjectReference {
        self.0
    }
    fn set_reference(&mut self, object: ObjectReference) {
        self.0 = object;
    }
    fn keep_alive<E: mmtk::scheduler::ProcessEdgesWork>(&mut self, _trace: &mut E) {}
}

impl ReferenceGlue<JuliaMlirVM> for JuliaMlirReferenceGlue {
    type FinalizableType = DummyFinalizable;

    fn set_referent(_reference: ObjectReference, _referent: ObjectReference) {}
    fn clear_referent(_new_reference: ObjectReference) {}
    fn get_referent(_object: ObjectReference) -> Option<ObjectReference> {
        None
    }
    fn enqueue_references(_references: &[ObjectReference], _tls: VMWorkerThread) {}
}

// ============================================================================
// Global state
// ============================================================================

static MMTK_INITIALIZED: AtomicBool = AtomicBool::new(false);

// Use raw static mut instead of lazy_static to avoid TLS issues in freestanding.
// Safety: jlmlir_gc_init is called exactly once before any other MMTk call.
static mut MMTK_INSTANCE: Option<Box<MMTK<JuliaMlirVM>>> = None;

fn get_mmtk() -> &'static MMTK<JuliaMlirVM> {
    unsafe { MMTK_INSTANCE.as_ref().expect("MMTk not initialized") }
}

// ============================================================================
// GC root stack globals — set by the MLIR-compiled code via jlmlir_gc_set_*
// ============================================================================

static GC_ROOT_STACK_BASE: AtomicUsize = AtomicUsize::new(0);
/// Address of the @_jlmlir_gc_root_top global in the MLIR binary.
/// Dereference to read the current root stack top value.
/// This avoids per-root FFI calls — the MLIR code writes root_top
/// directly to the global, and we read it through this indirection.
static GC_ROOT_STACK_TOP_ADDR: AtomicUsize = AtomicUsize::new(0);

// Global roots table (type descriptors, interned symbols, etc.)
static GLOBAL_ROOTS_BASE: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_ROOTS_COUNT: AtomicUsize = AtomicUsize::new(0);

// GC statistics
static GC_COUNT: AtomicUsize = AtomicUsize::new(0);

// Safepoint page address — set by jlmlir_gc_init, read by MLIR-compiled code
static SAFEPOINT_PAGE: AtomicUsize = AtomicUsize::new(0);

extern "C" {
    fn jlmlir_safepoint_init(gc_callback: extern "C" fn()) -> *mut u8;
    fn jlmlir_safepoint_arm();
    fn jlmlir_safepoint_disarm();
}

// ============================================================================
// C API — called by the MLIR-compiled freestanding binary
// ============================================================================

/// Initialize the MMTk heap with the specified plan and heap size.
///
/// Must be called exactly once before any allocation.
///
/// # Arguments
/// * `heap_size` - Maximum heap size in bytes.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_init(heap_size: usize) {
    assert!(
        !MMTK_INITIALIZED.load(Ordering::SeqCst),
        "jlmlir_gc_init called twice"
    );

    let mut builder = MMTKBuilder::new();

    // Select plan based on Cargo feature flags.
    use mmtk::util::options::PlanSelector;

    #[cfg(any(feature = "julia_mlir_immix", feature = "julia_mlir_immix_moving"))]
    {
        builder.options.plan.set(PlanSelector::Immix);
    }
    #[cfg(feature = "julia_mlir_lxr")]
    {
        builder.options.plan.set(PlanSelector::LXR);
    }
    #[cfg(all(
        not(feature = "julia_mlir_immix"),
        not(feature = "julia_mlir_immix_moving"),
        not(feature = "julia_mlir_lxr"),
    ))]
    {
        builder.options.plan.set(PlanSelector::NoGC);
    }

    let success =
        builder
            .options
            .gc_trigger
            .set(mmtk::util::options::GCTriggerSelector::FixedHeapSize(
                heap_size,
            ));
    assert!(success, "Failed to set heap size to {}", heap_size);

    // Single GC worker thread for N4a.
    let _ = builder.options.threads.set(1);

    let mmtk_box = memory_manager::mmtk_init(&builder);
    MMTK_INSTANCE = Some(mmtk_box);
    MMTK_INITIALIZED.store(true, Ordering::SeqCst);

    // Spawn GC worker thread(s) and enable collection.
    // This calls spawn_gc_thread which uses pthread_create from the libc shim.
    memory_manager::initialize_collection(
        get_mmtk(),
        VMThread(OpaquePointer::from_address(Address::from_usize(1))),
    );

    // Initialize page-protection safepoint.
    let page = jlmlir_safepoint_init(safepoint_gc_block);
    SAFEPOINT_PAGE.store(page as usize, Ordering::SeqCst);
}

/// Callback invoked by the SIGSEGV handler when the mutator faults on the
/// safepoint page. Flushes barrier buffers, blocks for GC, then returns
/// (the handler disarms the page so the faulting instruction re-executes).
extern "C" fn safepoint_gc_block() {
    // Flush barrier buffers before blocking.
    unsafe {
        let mutator_ptr = THE_MUTATOR.load(Ordering::SeqCst);
        if mutator_ptr != 0 {
            let mutator = &mut *(mutator_ptr as *mut Mutator<JuliaMlirVM>);
            mutator.barrier.flush();
        }
    }
    // Block for GC (sets MUTATOR_STOPPED, waits for GC_DONE).
    let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(unsafe {
        Address::from_usize(1)
    })));
    <JuliaMlirCollection as Collection<JuliaMlirVM>>::block_for_gc(tls);
}

/// Bind a mutator to the current thread and return a pointer to it.
///
/// Single-threaded: called once. The returned pointer must be passed
/// to all subsequent allocation calls.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_bind_mutator() -> *mut Mutator<JuliaMlirVM> {
    let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(Address::from_usize(
        1,
    ))));
    let mutator_box = memory_manager::bind_mutator(get_mmtk(), tls);
    let mutator_ptr = Box::into_raw(mutator_box);
    THE_MUTATOR.store(mutator_ptr as usize, Ordering::SeqCst);
    mutator_ptr
}

/// Allocate an object on the managed heap.
///
/// Returns a pointer to the object body. The caller must write the type tag
/// at (result - 8).
///
/// Internal layout: [tag 8B | body NB]
/// Returned pointer → body start.
///
/// # Arguments
/// * `mutator`   - From `jlmlir_bind_mutator`.
/// * `body_size` - Size of the body in bytes (not including tag word).
#[no_mangle]
pub unsafe extern "C" fn jlmlir_alloc(
    mutator: *mut Mutator<JuliaMlirVM>,
    body_size: usize,
) -> *mut u8 {
    // Total: [tag 8B | body]
    let total_size = 8 + body_size;
    let aligned_size = (total_size + 15) & !15;

    let alloc_start = memory_manager::alloc(
        &mut *mutator,
        aligned_size,
        16,
        0,
        AllocationSemantics::Default,
    );

    if alloc_start.is_zero() {
        return std::ptr::null_mut();
    }

    // ObjectReference = alloc_start + 8 (body start, after tag word)
    let obj_ref = ObjectReference::from_raw_address_unchecked(alloc_start + 8usize);

    memory_manager::post_alloc(
        &mut *mutator,
        obj_ref,
        aligned_size,
        AllocationSemantics::Default,
    );

    let body_addr: Address = alloc_start + 8usize;
    body_addr.to_mut_ptr::<u8>()
}

/// Allocate and set the type tag in one call.
///
/// Returns pointer to body start.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_alloc_typed(
    mutator: *mut Mutator<JuliaMlirVM>,
    body_size: usize,
    type_tag: usize,
) -> *mut u8 {
    let ptr = jlmlir_alloc(mutator, body_size);
    if !ptr.is_null() {
        // Write tag word at (ptr - 8)
        let tag_addr = (ptr as usize - 8) as *mut usize;
        *tag_addr = type_tag;
    }
    ptr
}

/// Read the type tag of an object (the DataType pointer).
#[no_mangle]
pub unsafe extern "C" fn jlmlir_typeof(obj: *const u8) -> usize {
    let tag_addr = (obj as usize - 8) as *const usize;
    *tag_addr
}

/// Get a pointer-sized field from an object at the given byte offset.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_getfield_raw(obj: *const u8, offset: usize) -> usize {
    let field_addr = (obj as usize + offset) as *const usize;
    *field_addr
}

/// Set a pointer-sized field on an object at the given byte offset.
/// For Immix (non-moving, N4a), no write barrier is needed.
/// For moving Immix (N4b) and LXR (N4c), write barriers are added.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_setfield_raw(obj: *mut u8, offset: usize, value: usize) {
    let field_addr = (obj as usize + offset) as *mut usize;
    *field_addr = value;
}

// ============================================================================
// Write barrier API for LXR (N4c)
//
// LXR uses a field-logging barrier. Before writing a new pointer value into
// a managed object's field, the mutator must call the barrier so that LXR can:
//   1. Read the old value from the slot (for RC decrement)
//   2. Record the new slot (for RC increment)
//   3. SATB-log the old value if concurrent marking is active
//
// The barrier checks a per-field "unlog bit" in side metadata.  If the field
// is already logged (common case), the check is a single byte load → no-op.
// Otherwise the slow path is taken.
//
// API contract:
//   jlmlir_write_barrier_pre(mutator, src_obj, slot_addr, new_target)
//     — Call BEFORE writing the pointer.
//     — src_obj: the object being mutated (ObjectReference = body start)
//     — slot_addr: address of the field being written
//     — new_target: the new pointer value being stored (0 if non-ref)
//
//   jlmlir_setfield_barrier(mutator, src_obj, offset, new_value)
//     — Subsuming barrier + store in one call (convenience wrapper).
//       Calls the pre-barrier, then performs the store.
// ============================================================================

/// Write barrier pre-call for LXR.  Call BEFORE writing a pointer into a
/// managed object field.  Under non-LXR plans this is a no-op.
///
/// # Arguments
/// * `mutator`    - Mutator pointer from `jlmlir_bind_mutator`.
/// * `src`        - The object being mutated (pointer to body start).
/// * `slot_addr`  - Address of the field about to be written.
/// * `new_target` - The new pointer value to be stored (body pointer, or 0).
#[no_mangle]
pub unsafe extern "C" fn jlmlir_write_barrier_pre(
    mutator: *mut Mutator<JuliaMlirVM>,
    src: *const u8,
    slot_addr: *mut u8,
    new_target: *const u8,
) {
    let src_addr = Address::from_usize(src as usize);
    if let Some(src_obj) = ObjectReference::from_raw_address(src_addr) {
        let slot = SimpleSlot::from_address(Address::from_usize(slot_addr as usize));
        let target = if new_target.is_null() {
            None
        } else {
            ObjectReference::from_raw_address(Address::from_usize(new_target as usize))
        };
        memory_manager::object_reference_write_pre(&mut *mutator, src_obj, slot, target);
    }
}

/// Subsuming write barrier + store.  Sets a pointer field on a managed object
/// with proper LXR write barrier semantics.
///
/// Equivalent to:
///   jlmlir_write_barrier_pre(mutator, obj, &obj[offset], value)
///   obj[offset] = value
///
/// # Arguments
/// * `mutator` - Mutator pointer from `jlmlir_bind_mutator`.
/// * `obj`     - The object being mutated (pointer to body start).
/// * `offset`  - Byte offset of the field within the body.
/// * `value`   - The new pointer-sized value to store.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_setfield_barrier(
    mutator: *mut Mutator<JuliaMlirVM>,
    obj: *mut u8,
    offset: usize,
    value: usize,
) {
    let slot_addr = (obj as usize + offset) as *mut u8;
    let new_target = value as *const u8;
    jlmlir_write_barrier_pre(mutator, obj, slot_addr, new_target);
    // Perform the actual store
    let field_addr = slot_addr as *mut usize;
    *field_addr = value;
}

/// Flush the mutator's barrier buffers.  Should be called at safepoints
/// and before GC to ensure all pending RC increments/decrements and SATB
/// entries are published to the GC workers.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_barrier_flush(mutator: *mut Mutator<JuliaMlirVM>) {
    (*mutator).barrier.flush();
}

/// Get total used bytes.
#[no_mangle]
pub extern "C" fn jlmlir_used_bytes() -> usize {
    memory_manager::used_bytes(get_mmtk())
}

/// Get total heap capacity.
#[no_mangle]
pub extern "C" fn jlmlir_total_bytes() -> usize {
    memory_manager::total_bytes(get_mmtk())
}

// ============================================================================
// New N4 C API — GC root management and safepoints
// ============================================================================

/// Set the GC root stack pointers. Called once at startup by the MLIR binary.
/// The root stack is a flat array of pointer-sized slots, growing upward.
/// `base` = start of the root stack array.
/// `top_addr` = address of the @_jlmlir_gc_root_top global variable.
///              The GC reads `*(top_addr)` during root scanning to get the
///              current top value. This eliminates per-root FFI calls —
///              the MLIR code writes root_top directly to the global.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_set_root_stack(base: usize, top_addr: usize) {
    GC_ROOT_STACK_BASE.store(base, Ordering::SeqCst);
    GC_ROOT_STACK_TOP_ADDR.store(top_addr, Ordering::SeqCst);
}

/// Register a global root slot. The GC will scan this slot during collection.
/// `slot_addr` is the address of a pointer-sized slot containing a managed reference.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_add_global_root(slot_addr: usize) {
    // Simple approach: maintain a dynamically-growing array of global root slots.
    // For N4a, we pre-allocate space and just bump a counter.
    let idx = GLOBAL_ROOTS_COUNT.fetch_add(1, Ordering::SeqCst);
    let base = GLOBAL_ROOTS_BASE.load(Ordering::Relaxed);
    if base != 0 {
        let entry_addr = base + idx * 8;
        *(entry_addr as *mut usize) = slot_addr;
    }
}

/// Initialize the global roots table with a given capacity.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_init_global_roots(table_addr: usize, _capacity: usize) {
    GLOBAL_ROOTS_BASE.store(table_addr, Ordering::SeqCst);
    GLOBAL_ROOTS_COUNT.store(0, Ordering::SeqCst);
}

/// Return the address of the safepoint page for volatile loads.
/// The MLIR lowering stores this in a global and emits `volatile load i32`
/// from it at each safepoint. When GC is requested, the page is mprotected
/// to PROT_NONE, causing a SIGSEGV that the libc shim handler catches.
#[no_mangle]
pub extern "C" fn jlmlir_safepoint_page() -> usize {
    SAFEPOINT_PAGE.load(Ordering::Relaxed)
}

/// Return the absolute base address of the LXR field-unlog-bit side metadata.
///
/// The MLIR lowering inlines the write barrier fast path as:
///   meta_byte = load_byte(base + (slot_addr >> 6))
///   bit_pos   = (slot_addr >> 3) & 7
///   if (meta_byte & (1 << bit_pos)) == 0: skip   // already logged
///   else: call slow path
///
/// The base address is computed from the FIELD_UNLOG_SIDE_METADATA_SPEC's
/// absolute offset in the contiguous global side metadata space.
///
/// Returns 0 when the LXR plan is not active (no field barrier needed).
#[no_mangle]
pub extern "C" fn jlmlir_field_unlog_bit_base_address() -> usize {
    let spec = FIELD_UNLOG_SIDE_METADATA_SPEC.as_spec();
    match spec {
        mmtk::util::metadata::MetadataSpec::OnSide(side_spec) => {
            side_spec.get_absolute_offset().as_usize()
        }
        _ => 0,
    }
}

/// Return the log_bytes_in_region for the field unlog bit spec.
/// This is the right-shift applied to the data address before indexing
/// into the metadata table. For the field-unlog-bit, this is
/// LOG_BYTES_IN_ADDRESS (= 3 on 64-bit), meaning 1 bit per pointer-sized word.
///
/// The metadata byte address is: base + (data_addr >> (log_bytes_in_region + 3))
/// The bit position within that byte: (data_addr >> log_bytes_in_region) & 7
#[no_mangle]
pub extern "C" fn jlmlir_field_unlog_bit_log_region() -> usize {
    let spec = FIELD_UNLOG_SIDE_METADATA_SPEC.as_spec();
    match spec {
        mmtk::util::metadata::MetadataSpec::OnSide(side_spec) => side_spec.log_bytes_in_region,
        _ => 0,
    }
}

/// GC safepoint check. Called by the MLIR code at loop back-edges and
/// function prologues. If GC is requested, blocks until collection completes.
///
/// Returns 1 if GC occurred, 0 otherwise.
///
/// Legacy function-call safepoint — kept for backward compatibility.
/// New code uses the page-protection safepoint via volatile load.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_safepoint() -> u32 {
    if GC_REQUESTED.load(Ordering::Relaxed) {
        // Flush barrier buffers before stopping — LXR needs pending
        // RC inc/dec/SATB entries published before collection starts.
        let mutator_ptr = THE_MUTATOR.load(Ordering::SeqCst);
        if mutator_ptr != 0 {
            let mutator = &mut *(mutator_ptr as *mut Mutator<JuliaMlirVM>);
            mutator.barrier.flush();
        }
        // We're at a safepoint — block for GC.
        let tls = VMMutatorThread(VMThread(OpaquePointer::from_address(Address::from_usize(
            1,
        ))));
        <JuliaMlirCollection as Collection<JuliaMlirVM>>::block_for_gc(tls);
        GC_COUNT.fetch_add(1, Ordering::Relaxed);
        1
    } else {
        0
    }
}

/// Get the number of GC collections that have occurred.
#[no_mangle]
pub extern "C" fn jlmlir_gc_count() -> usize {
    GC_COUNT.load(Ordering::Relaxed)
}

/// Explicitly trigger a GC collection. For testing.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_collect() {
    memory_manager::handle_user_collection_request(
        get_mmtk(),
        VMMutatorThread(VMThread(OpaquePointer::from_address(Address::from_usize(
            1,
        )))),
        true, // force
    );
}
