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
const TYPEDESC_OFF_FIELD_TYPES: usize = 48;

const TYPE_FLAG_ABSTRACT: u64 = 0x1;

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

    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec =
        VMLocalForwardingBitsSpec::side_after(MARKING_METADATA_SPEC.as_spec());

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
        // Read instance_size from the type descriptor.
        // tag_word at (obj - 8), type descriptor at *tag_word,
        // instance_size at type_desc + 8.
        unsafe {
            let tag_addr = object.to_raw_address() - 8usize;
            let type_desc = Address::from_usize(tag_addr.load::<usize>());
            if type_desc.is_zero() {
                // Defensive: no type descriptor means we can't determine size.
                // Return minimum object size (tag + 16-byte align).
                return 16;
            }
            let instance_size = (type_desc + TYPEDESC_OFF_INSTANCE_SZ).load::<u64>() as usize;
            // Total allocation size: tag_word (8B) + body, aligned to 16
            let total = 8 + instance_size;
            (total + 15) & !15
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

pub struct JuliaMlirScanning;

impl Scanning<JuliaMlirVM> for JuliaMlirScanning {
    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        _mutator: &'static mut Mutator<JuliaMlirVM>,
        mut factory: impl RootsWorkFactory<SimpleSlot>,
    ) {
        // Enumerate roots from the GC root stack.
        // The MLIR-compiled code maintains a shadow stack at a well-known global:
        //   _jlmlir_gc_root_stack_top -> pointer to top of root stack
        //   _jlmlir_gc_root_stack_base -> pointer to base of root stack
        // Each entry on the stack is a pointer to a heap object (8 bytes).
        unsafe {
            let top_ptr = GC_ROOT_STACK_TOP.load(Ordering::Relaxed);
            let base_ptr = GC_ROOT_STACK_BASE.load(Ordering::Relaxed);
            if top_ptr == 0 || base_ptr == 0 || top_ptr <= base_ptr {
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
        // Read the type descriptor to determine which fields are pointers.
        // For N4a, all fields in the body are treated as potential pointers
        // (conservative for correctness). The type descriptor's nfields
        // and field_types guide precise scanning.
        unsafe {
            let obj_addr = object.to_raw_address();
            let tag_addr = obj_addr - 8usize;
            let type_desc_addr = Address::from_usize(tag_addr.load::<usize>());
            if type_desc_addr.is_zero() {
                return;
            }

            let flags = (type_desc_addr + TYPEDESC_OFF_FLAGS).load::<u64>();
            if flags & TYPE_FLAG_ABSTRACT != 0 {
                // Abstract types have no instances — shouldn't be on the heap.
                return;
            }

            let nfields = (type_desc_addr + TYPEDESC_OFF_NFIELDS).load::<u64>() as usize;
            let field_types_ptr_val = (type_desc_addr + TYPEDESC_OFF_FIELD_TYPES).load::<usize>();

            if nfields == 0 {
                // No fields to scan (e.g., Int64, Float64, Bool, Nothing).
                return;
            }

            if field_types_ptr_val != 0 {
                // Precise scanning: use field_types to determine which fields are pointers.
                // field_types is an array of type descriptor pointers.
                // A field is a pointer if its type descriptor is non-null and the
                // instance_size of that field's type is 0 (abstract/box) or > 0 (concrete heap type).
                // For now, scan all fields conservatively — every 8-byte field
                // that looks like a managed pointer is reported.
                // TODO(N4b): Use field_types for precise per-field scanning
                for i in 0..nfields {
                    let field_addr = obj_addr + (i * 8);
                    let slot = SimpleSlot::from_address(field_addr);
                    slot_visitor.visit_slot(slot, false);
                }
            } else {
                // No field type info — conservatively scan all pointer-sized fields.
                let instance_size =
                    (type_desc_addr + TYPEDESC_OFF_INSTANCE_SZ).load::<u64>() as usize;
                let num_words = instance_size / 8;
                for i in 0..num_words {
                    let field_addr = obj_addr + (i * 8);
                    let slot = SimpleSlot::from_address(field_addr);
                    slot_visitor.visit_slot(slot, false);
                }
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
            let top = GC_ROOT_STACK_TOP.load(Ordering::SeqCst);
            let base = GC_ROOT_STACK_BASE.load(Ordering::SeqCst);
            if top == 0 || base == 0 || top <= base {
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
static GC_ROOT_STACK_TOP: AtomicUsize = AtomicUsize::new(0);

// Global roots table (type descriptors, interned symbols, etc.)
static GLOBAL_ROOTS_BASE: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_ROOTS_COUNT: AtomicUsize = AtomicUsize::new(0);

// GC statistics
static GC_COUNT: AtomicUsize = AtomicUsize::new(0);

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
/// `base` = start of the array, `top` = pointer to current top variable.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_set_root_stack(base: usize, top_ptr: usize) {
    GC_ROOT_STACK_BASE.store(base, Ordering::SeqCst);
    // top_ptr is the address of the variable holding the current top.
    // We need to read through this indirection during root scanning.
    GC_ROOT_STACK_TOP.store(top_ptr, Ordering::SeqCst);
}

/// Update the root stack top. Called by the MLIR code at each safepoint
/// to publish the current stack height to the GC.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_update_root_top(top: usize) {
    GC_ROOT_STACK_TOP.store(top, Ordering::SeqCst);
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

/// GC safepoint check. Called by the MLIR code at loop back-edges and
/// function prologues. If GC is requested, blocks until collection completes.
///
/// Returns 1 if GC occurred, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn jlmlir_gc_safepoint() -> u32 {
    if GC_REQUESTED.load(Ordering::Relaxed) {
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
