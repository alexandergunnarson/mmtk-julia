// All functions here are extern function. There is no point for marking them as unsafe.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
use crate::JuliaVM;
use crate::JULIA_HEADER_SIZE;
use crate::SINGLETON;
use crate::{BUILDER, DISABLED_GC, MUTATORS, USER_TRIGGERED_GC};

use libc::c_char;
use log::*;
use mmtk::memory_manager;
use mmtk::scheduler::GCWorker;
use mmtk::util::api_util::NullableObjectReference;
use mmtk::util::opaque_pointer::*;
use mmtk::util::{Address, ObjectReference, OpaquePointer};
use mmtk::AllocationSemantics;
use mmtk::Mutator;
use std::ffi::CStr;
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// Helper: downcast the mutator's barrier to access LXR field barrier
// semantics (inc/dec queues).  Used by non-heap write barrier and explicit
// RC dec FFI functions to push entries through the mutator's own queues
// instead of thread-local buffers.  The mutator's queues are flushed by
// Mutator::flush() during STW (called on every mutator thread), which is
// the architecturally correct flush point.
#[inline]
fn get_lxr_semantics(
    mutator: &mut Mutator<JuliaVM>,
) -> &mut mmtk::plan::lxr::barrier::LXRFieldBarrierSemantics<JuliaVM> {
    use mmtk::plan::barriers::FieldBarrier;
    use mmtk::plan::lxr::barrier::LXRFieldBarrierSemantics;
    use mmtk::MutatorContext;
    let barrier: &mut dyn mmtk::plan::barriers::Barrier<JuliaVM> = mutator.barrier();
    let field_barrier: &mut FieldBarrier<LXRFieldBarrierSemantics<JuliaVM>> = barrier
        .downcast_mut()
        .expect("LXR plan must use FieldBarrier<LXRFieldBarrierSemantics>");
    &mut field_barrier.semantics
}

#[no_mangle]
pub extern "C" fn mmtk_gc_init(
    min_heap_size: usize,
    max_heap_size: usize,
    n_gcthreads: usize,
    header_size: usize,
    buffer_tag: usize,
) {
    unsafe {
        crate::JULIA_HEADER_SIZE = header_size;
        crate::JULIA_BUFF_TAG = buffer_tag;
    };

    {
        let mut builder = BUILDER.lock().unwrap();

        // Set plan
        use mmtk::util::options::PlanSelector;
        let force_plan = if cfg!(feature = "nogc") {
            Some(PlanSelector::NoGC)
        } else if cfg!(feature = "marksweep") {
            Some(PlanSelector::MarkSweep)
        } else if cfg!(feature = "immix") {
            Some(PlanSelector::Immix)
        } else if cfg!(feature = "stickyimmix") {
            Some(PlanSelector::StickyImmix)
        } else if cfg!(feature = "lxr") {
            Some(PlanSelector::LXR)
        } else {
            None
        };
        if let Some(plan) = force_plan {
            builder.options.plan.set(plan);
        }

        // Set heap size
        let success =
            // By default min and max heap size are 0, and we use the Stock GC heuristics
            if min_heap_size == 0 && max_heap_size == 0 {
                info!(
                    "Setting mmtk heap size to use Stock GC heuristics as defined in gc_trigger.rs",
                );
                builder
                    .options
                    .gc_trigger
                    .set(mmtk::util::options::GCTriggerSelector::Delegated)
            } else if min_heap_size != 0 {
                info!(
                    "Setting mmtk heap size to a variable size with min-max of {}-{} (in bytes)",
                    min_heap_size, max_heap_size
                );
                builder.options.gc_trigger.set(
                    mmtk::util::options::GCTriggerSelector::DynamicHeapSize(
                        min_heap_size,
                        max_heap_size,
                    ),
                )
            } else {
                info!(
                    "Setting mmtk heap size to a fixed max of {} (in bytes)",
                    max_heap_size
                );
                builder.options.gc_trigger.set(
                    mmtk::util::options::GCTriggerSelector::FixedHeapSize(max_heap_size),
                )
            };
        assert!(
            success,
            "Failed to set heap size to {}-{}",
            min_heap_size, max_heap_size
        );

        // Set using weak references
        let success = builder.options.no_reference_types.set(false);
        assert!(success, "Failed to set no_reference_types to false");

        // Set GC threads
        if n_gcthreads > 0 {
            let success = builder.options.threads.set(n_gcthreads);
            assert!(success, "Failed to set GC threads to {}", n_gcthreads);
        }
    }

    // Make sure that we haven't initialized MMTk (by accident) yet
    assert!(!crate::MMTK_INITIALIZED.load(Ordering::SeqCst));
    // Make sure we initialize MMTk here
    lazy_static::initialize(&SINGLETON);

    // Runtime-initialize the heap range constants used by the C write-barrier
    // fast path.  vm_layout() is valid now that MMTk is initialized.
    // This must happen before any write barrier fires (i.e. before the first
    // allocation that stores a pointer into a heap field).
    //
    // The range must cover exactly the contiguous spaces that have field unlog
    // bit metadata committed: Immix (space 0), Immortal (space 1), LOS
    // (space 2) — 3 spaces of max_space_extent each.  Using the wider
    // vm_layout().heap_end (0x2200...) would accept addresses in uncommitted
    // metadata regions between the last space and the end of the potential
    // address space, causing SIGSEGV on unlog bit reads.
    {
        use mmtk::util::heap::layout::vm_layout::vm_layout;
        let layout = vm_layout();
        let start = layout.heap_start.as_usize();
        let extent = layout.max_space_extent();
        MMTK_HEAP_START.store(start, Ordering::Relaxed);
        // 3 contiguous spaces: Immix + Immortal + LOS
        MMTK_HEAP_END.store(start + 3 * extent, Ordering::Relaxed);
        #[cfg(feature = "lxr_rc_trace")]
        eprintln!(
            "[rc-trace heap-range] MMTK_HEAP_START={:#x} MMTK_HEAP_END={:#x} extent={:#x}",
            start,
            start + 3 * extent,
            extent,
        );
    }

    // Initialize GC timing infrastructure.  The LXR fork's Timer uses
    // Option<Instant> which panics on unwrap if not initialized.
    // ScheduleCollection sets these, but report_gc_start may read them
    // first during a user-triggered GC.gc().
    mmtk::GC_TRIGGER_TIME.start();
    mmtk::GC_START_TIME.start();

    // Initialize per-object RC tracing from MMTK_RC_TRACE_ADDRS env var.
    // This must happen after MMTk is initialized (so the trace statics exist)
    // but before any GC work starts.  The env var format is comma-separated
    // hex addresses, e.g. "0x200ffc01000,0x200ffc02000".
    // Build with --features lxr_rc_trace to enable.
    #[cfg(feature = "lxr_rc_trace")]
    mmtk::plan::lxr::rc::rc_trace_init_from_env();

    // Initialize the death-time genuine-undercount detector (§2.17) from
    // MMTK_RC_UNDERCOUNT_BUDGET / MMTK_RC_UNDERCOUNT_FROM_GC.  When enabled, it
    // reports any freed object that still has a LIVE heap referrer (the genuine
    // RC undercount), excluding the proven-benign matches=0 deaths (§2.16).
    #[cfg(feature = "lxr_rc_trace")]
    mmtk::plan::lxr::rc::rc_undercount_init_from_env();

    // Hijack the panic hook to make sure that if we crash in the GC threads, the process aborts.
    crate::set_panic_hook();

    // Assert to make sure our fastpath allocation is correct.
    {
        // If the assertion failed, check the allocation fastpath in Julia
        // - runtime fastpath: mmtk_immix_alloc_fast and mmtk_immortal_alloc_fast in julia.h
        // - compiler inserted fastpath: llvm-final-gc-lowering.cpp
        use mmtk::util::alloc::AllocatorSelector;
        let default_allocator = memory_manager::get_allocator_mapping::<JuliaVM>(
            &SINGLETON,
            AllocationSemantics::Default,
        );
        assert_eq!(default_allocator, AllocatorSelector::Immix(0));
        let immortal_allocator = memory_manager::get_allocator_mapping::<JuliaVM>(
            &SINGLETON,
            AllocationSemantics::Immortal,
        );
        assert_eq!(immortal_allocator, AllocatorSelector::BumpPointer(0));
    }

    // Assert to make sure alignment used in C is correct
    {
        // If the assertion failed, check MMTK_MIN_ALIGNMENT in julia.h
        assert_eq!(<JuliaVM as mmtk::vm::VMBinding>::MIN_ALIGNMENT, 4);
    }
}

#[no_mangle]
pub extern "C" fn mmtk_bind_mutator(tls: VMMutatorThread, tid: usize) -> *mut Mutator<JuliaVM> {
    let mutator_box = memory_manager::bind_mutator(&SINGLETON, tls);

    let res = Box::into_raw(mutator_box);

    info!("Binding mutator {:?} to thread id = {}", res, tid);
    res
}

#[no_mangle]
pub extern "C" fn mmtk_post_bind_mutator(
    mutator: *mut Mutator<JuliaVM>,
    original_box_mutator: *mut Mutator<JuliaVM>,
) {
    // We have to store the original boxed mutator. Otherwise, we may have dangling pointers in mutator.
    MUTATORS.write().unwrap().insert(
        Address::from_mut_ptr(mutator),
        Address::from_mut_ptr(original_box_mutator),
    );
}

#[no_mangle]
pub extern "C" fn mmtk_destroy_mutator(mutator: *mut Mutator<JuliaVM>) {
    // destroy the mutator with MMTk.
    memory_manager::destroy_mutator(unsafe { &mut *mutator });

    let mut mutators = MUTATORS.write().unwrap();
    let key = Address::from_mut_ptr(mutator);

    // Clear the original boxed mutator
    let orig_mutator = mutators.get(&key).unwrap();
    let _ = unsafe { Box::from_raw(orig_mutator.to_mut_ptr::<Mutator<JuliaVM>>()) };

    // Remove from our hashmap
    mutators.remove(&key);
}

#[no_mangle]
pub extern "C" fn mmtk_alloc(
    mutator: *mut Mutator<JuliaVM>,
    size: usize,
    align: usize,
    offset: usize,
    semantics: AllocationSemantics,
) -> Address {
    debug_assert!(
        mmtk::util::conversions::raw_is_aligned(
            size,
            <JuliaVM as mmtk::vm::VMBinding>::MIN_ALIGNMENT
        ),
        "Alloc size {} is not aligned to min alignment",
        size
    );
    memory_manager::alloc::<JuliaVM>(unsafe { &mut *mutator }, size, align, offset, semantics)
}

#[no_mangle]
pub extern "C" fn mmtk_alloc_large(
    mutator: *mut Mutator<JuliaVM>,
    size: usize,
    align: usize,
    offset: usize,
) -> Address {
    memory_manager::alloc::<JuliaVM>(
        unsafe { &mut *mutator },
        size,
        align,
        offset,
        AllocationSemantics::Los,
    )
}

#[no_mangle]
pub extern "C" fn mmtk_post_alloc(
    mutator: *mut Mutator<JuliaVM>,
    refer: ObjectReference,
    bytes: usize,
    semantics: AllocationSemantics,
) {
    memory_manager::post_alloc::<JuliaVM>(unsafe { &mut *mutator }, refer, bytes, semantics)
}

#[no_mangle]
pub extern "C" fn mmtk_will_never_move(object: ObjectReference) -> bool {
    !object.is_movable()
}

#[no_mangle]
pub extern "C" fn mmtk_start_worker(tls: VMWorkerThread, worker: *mut GCWorker<JuliaVM>) {
    let worker = unsafe { Box::from_raw(worker) };
    memory_manager::start_worker::<JuliaVM>(&SINGLETON, tls, worker)
}

#[no_mangle]
pub extern "C" fn mmtk_initialize_collection(tls: VMThread) {
    memory_manager::initialize_collection(&SINGLETON, tls);
}

#[no_mangle]
pub extern "C" fn mmtk_used_bytes() -> usize {
    memory_manager::used_bytes(&SINGLETON)
}

#[no_mangle]
pub extern "C" fn mmtk_free_bytes() -> usize {
    memory_manager::free_bytes(&SINGLETON)
}

#[no_mangle]
pub extern "C" fn mmtk_total_bytes() -> usize {
    memory_manager::total_bytes(&SINGLETON)
}

#[no_mangle]
pub extern "C" fn mmtk_is_live_object(object: ObjectReference) -> bool {
    object.is_live()
}

#[no_mangle]
pub extern "C" fn mmtk_is_mapped_address(address: Address) -> bool {
    address.is_mapped()
}

#[no_mangle]
pub extern "C" fn mmtk_handle_user_collection_request(tls: VMMutatorThread, collection: u8) {
    AtomicIsize::fetch_add(&USER_TRIGGERED_GC, 1, Ordering::SeqCst);
    if AtomicBool::load(&DISABLED_GC, Ordering::SeqCst) {
        AtomicIsize::fetch_add(&USER_TRIGGERED_GC, -1, Ordering::SeqCst);
        return;
    }
    // See jl_gc_collection_t
    match collection {
        // auto
        0 => memory_manager::handle_user_collection_request::<JuliaVM>(&SINGLETON, tls, false),
        // full
        1 => SINGLETON.handle_user_collection_request(tls, true, true),
        // incremental
        2 => SINGLETON.handle_user_collection_request(tls, true, false),
        _ => unreachable!(),
    };
}

#[no_mangle]
pub extern "C" fn mmtk_add_weak_candidate(reff: ObjectReference) {
    memory_manager::add_weak_candidate(&SINGLETON, reff)
}

#[no_mangle]
pub extern "C" fn mmtk_add_soft_candidate(reff: ObjectReference) {
    memory_manager::add_soft_candidate(&SINGLETON, reff)
}

#[no_mangle]
pub extern "C" fn mmtk_add_phantom_candidate(reff: ObjectReference) {
    memory_manager::add_phantom_candidate(&SINGLETON, reff)
}

#[no_mangle]
pub extern "C" fn mmtk_harness_begin(tls: VMMutatorThread) {
    memory_manager::harness_begin(&SINGLETON, tls)
}

#[no_mangle]
pub extern "C" fn mmtk_harness_end(_tls: OpaquePointer) {
    memory_manager::harness_end(&SINGLETON)
}

#[no_mangle]
pub extern "C" fn mmtk_process(name: *const c_char, value: *const c_char) -> bool {
    let name_str: &CStr = unsafe { CStr::from_ptr(name) };
    let value_str: &CStr = unsafe { CStr::from_ptr(value) };
    let mut builder = BUILDER.lock().unwrap();
    memory_manager::process(
        &mut builder,
        name_str.to_str().unwrap(),
        value_str.to_str().unwrap(),
    )
}

#[no_mangle]
pub extern "C" fn mmtk_starting_heap_address() -> Address {
    memory_manager::starting_heap_address()
}

#[no_mangle]
pub extern "C" fn mmtk_last_heap_address() -> Address {
    memory_manager::last_heap_address()
}

// Accessed from C to count the bytes we allocated with jl_gc_counted_malloc etc.
#[no_mangle]
pub static JULIA_MALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "C" fn mmtk_gc_poll(tls: VMMutatorThread) {
    memory_manager::gc_poll(&SINGLETON, tls);
}

#[no_mangle]
pub extern "C" fn mmtk_runtime_panic() {
    panic!("Panicking at runtime!")
}

#[no_mangle]
pub extern "C" fn mmtk_unreachable() {
    unreachable!()
}

#[no_mangle]
#[allow(mutable_transmutes)]
pub extern "C" fn mmtk_set_vm_space(start: Address, size: usize) {
    let mmtk: &mmtk::MMTK<JuliaVM> = &SINGLETON;
    let mmtk_mut: &mut mmtk::MMTK<JuliaVM> = unsafe { std::mem::transmute(mmtk) };
    memory_manager::set_vm_space(mmtk_mut, start, size);

    #[cfg(feature = "stickyimmix")]
    set_side_log_bit_for_region(start, size);
}

#[no_mangle]
pub extern "C" fn mmtk_memory_region_copy(
    mutator: *mut Mutator<JuliaVM>,
    src_obj: ObjectReference,
    src_addr: Address,
    dst_obj: ObjectReference,
    dst_addr: Address,
    count: usize,
) {
    use crate::slots::JuliaMemorySlice;
    let src = JuliaMemorySlice {
        owner: src_obj,
        start: src_addr,
        count,
    };
    let dst = JuliaMemorySlice {
        owner: dst_obj,
        start: dst_addr,
        count,
    };
    let mutator = unsafe { &mut *mutator };
    memory_manager::memory_region_copy(mutator, src, dst);
}

#[no_mangle]
#[allow(unused_variables)] // Args are only used for sticky immix.
pub extern "C" fn mmtk_immortal_region_post_alloc(start: Address, size: usize) {
    #[cfg(feature = "stickyimmix")]
    set_side_log_bit_for_region(start, size);
}

#[cfg(feature = "stickyimmix")]
fn set_side_log_bit_for_region(start: Address, size: usize) {
    debug!("Bulk set {} to {} ({} bytes)", start, start + size, size);
    use crate::mmtk::vm::ObjectModel;
    match <JuliaVM as mmtk::vm::VMBinding>::VMObjectModel::GLOBAL_LOG_BIT_SPEC.as_spec() {
        mmtk::util::metadata::MetadataSpec::OnSide(side) => side.bset_metadata(start, size),
        _ => unimplemented!(),
    }
}

#[no_mangle]
pub extern "C" fn mmtk_object_reference_write_post(
    mutator: *mut Mutator<JuliaVM>,
    src: ObjectReference,
    target: NullableObjectReference,
) {
    let mutator = unsafe { &mut *mutator };
    memory_manager::object_reference_write_post(
        mutator,
        src,
        crate::slots::JuliaVMSlot::Simple(mmtk::vm::slot::SimpleSlot::from_address(Address::ZERO)),
        target.into(),
    )
}

#[no_mangle]
pub extern "C" fn mmtk_object_reference_write_slow(
    mutator: &'static mut Mutator<JuliaVM>,
    src: ObjectReference,
    _target: NullableObjectReference,
) {
    use mmtk::MutatorContext;
    // The C-side post-write barrier (jl_gc_wb / mmtk_gc_wb_fast) does not
    // pass the slot address — only the parent object and written value.
    // LXR's field barrier (enqueue_node) requires a real slot address to
    // read/write the per-field unlog bits.  Passing Address::ZERO crashes
    // with SIGSEGV when the barrier tries to load metadata at address 0.
    //
    // Instead, use the object-level barrier: object_probable_write_slow
    // iterates ALL pointer fields of `src`, enqueuing each field's slot
    // with its actual address.  This is more expensive than a single-slot
    // barrier but is correct without slot info.
    //
    // For the fast path (field already logged), this is a no-op per slot.
    // For code paths that DO know the slot address (e.g. MLIR-compiled
    // code, C runtime pre-barriers), use mmtk_object_reference_write_pre
    // or jl_gc_wb_field_pre instead.
    mutator.barrier().object_probable_write(src);
}

/// Side log bit is the first side metadata spec starting.
#[no_mangle]
pub static MMTK_SIDE_LOG_BIT_BASE_ADDRESS: Address =
    mmtk::util::metadata::side_metadata::GLOBAL_SIDE_METADATA_VM_BASE_ADDRESS;

/// Heap range for the C write-barrier fast path.
///
/// Runtime-initialized from `vm_layout()` during `mmtk_gc_init`, after MMTk
/// is fully set up.  The range [MMTK_HEAP_START, MMTK_HEAP_END) covers
/// exactly the 3 contiguous spaces (Immix, Immortal, LOS) computed as
/// `heap_start + 3 * max_space_extent`.  This is tight: any address in this
/// range has field unlog bit metadata committed; any address outside does not.
///
/// Slots outside this range (malloc'd GenericMemory data buffers, C stack,
/// etc.) must NOT be checked via `_mmtk_check_bit`.
///
/// These are `AtomicUsize` so Rust can write them once during init while C
/// reads them on every barrier call.  On x86-64, `AtomicUsize` has identical
/// size/alignment to `usize`; C reads them as plain `uintptr_t` loads.
/// After init the values never change, so no synchronization overhead.
#[no_mangle]
pub static MMTK_HEAP_START: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub static MMTK_HEAP_END: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "C" fn mmtk_object_is_managed_by_mmtk(addr: usize) -> bool {
    crate::api::mmtk_is_mapped_address(unsafe { Address::from_usize(addr) })
}

#[no_mangle]
pub extern "C" fn mmtk_start_spawned_worker_thread(
    tls: VMWorkerThread,
    ctx: *mut GCWorker<JuliaVM>,
) {
    mmtk_start_worker(tls, ctx);
}

#[inline(always)]
pub fn store_obj_size(obj: ObjectReference, size: usize) {
    let addr_size = obj.to_raw_address() - 16;
    unsafe {
        addr_size.store::<u64>(size as u64);
    }
}

#[no_mangle]
pub extern "C" fn mmtk_store_obj_size_c(obj: ObjectReference, size: usize) {
    let addr_size = obj.to_raw_address() - 16;
    unsafe {
        addr_size.store::<u64>(size as u64);
    }
}

#[no_mangle]
pub extern "C" fn mmtk_get_obj_size(obj: ObjectReference) -> usize {
    unsafe {
        let addr_size = obj.to_raw_address() - 2 * JULIA_HEADER_SIZE;
        addr_size.load::<u64>() as usize
    }
}

#[cfg(all(feature = "object_pinning", not(feature = "non_moving")))]
#[no_mangle]
pub extern "C" fn mmtk_pin_object(object: ObjectReference) -> bool {
    // We may in the future replace this with a check for the immix space (bound check), which should be much cheaper.
    if mmtk_object_is_managed_by_mmtk(object.to_raw_address().as_usize()) {
        memory_manager::pin_object(object)
    } else {
        debug!("Object is not managed by mmtk - (un)pinning it via this function isn't supported.");
        false
    }
}

#[cfg(all(feature = "object_pinning", not(feature = "non_moving")))]
#[no_mangle]
pub extern "C" fn mmtk_unpin_object(object: ObjectReference) -> bool {
    if mmtk_object_is_managed_by_mmtk(object.to_raw_address().as_usize()) {
        memory_manager::unpin_object(object)
    } else {
        debug!("Object is not managed by mmtk - (un)pinning it via this function isn't supported.");
        false
    }
}

#[cfg(all(feature = "object_pinning", not(feature = "non_moving")))]
#[no_mangle]
pub extern "C" fn mmtk_is_pinned(object: ObjectReference) -> bool {
    if mmtk_object_is_managed_by_mmtk(object.to_raw_address().as_usize()) {
        memory_manager::is_pinned(object)
    } else {
        debug!("Object is not managed by mmtk - checking via this function isn't supported.");
        false
    }
}

// If the `non-moving` feature is selected, pinning/unpinning is a noop and simply returns false
#[cfg(all(feature = "object_pinning", feature = "non_moving"))]
#[no_mangle]
pub extern "C" fn mmtk_pin_object(_object: ObjectReference) -> bool {
    false
}

#[cfg(all(feature = "object_pinning", feature = "non_moving"))]
#[no_mangle]
pub extern "C" fn mmtk_unpin_object(_object: ObjectReference) -> bool {
    false
}

#[cfg(all(feature = "object_pinning", feature = "non_moving"))]
#[no_mangle]
pub extern "C" fn mmtk_is_pinned(_object: ObjectReference) -> bool {
    false
}

#[no_mangle]
pub extern "C" fn get_mmtk_version() -> *const c_char {
    crate::build_info::MMTK_JULIA_FULL_VERSION_STRING
        .as_c_str()
        .as_ptr() as _
}

// ====== LXR-specific FFI exports ======

/// LXR dec buffer: add an object to the decrement buffer for lazy RC processing.
/// Called by the MLIR compiler for RC decrements (julia.rc.dec ops).
///
/// In Perceus-style RC, a decrement happens when a variable goes out of scope.
/// The object is added to a thread-local buffer. When the buffer is full or at
/// GC safepoints, the buffer is flushed as a ProcessDecs work packet that is
/// either processed lazily (concurrent) or at STW (synchronous).
///
/// Under non-LXR plans, this is a no-op since tracing GCs don't use RC.
#[no_mangle]
pub extern "C" fn jl_gc_mmtk_dec_buf_add(mutator: *mut Mutator<JuliaVM>, obj: ObjectReference) {
    // Push the decrement directly into the mutator's barrier dec queue.
    // The barrier's dec queue is flushed by Mutator::flush() during STW
    // (called on every mutator thread in StopMutators::do_work), so all
    // pending decrements are processed before the GC sweeps.
    let mutator = unsafe { &mut *mutator };
    let semantics = get_lxr_semantics(mutator);
    semantics.push_dec(obj);
}

/// No-op: retained for FFI compatibility.
///
/// Prior to Phase 6.16, this flushed thread-local DEC_BUFFER and
/// NONHEAP_INC_BUFFER.  Those buffers no longer exist — all inc/dec
/// entries are now pushed directly into the mutator's barrier queues,
/// which are flushed by Mutator::flush() during STW.  This function
/// is still declared as an extern in the C runtime (gc-mmtk.c) so it
/// must exist, but it does nothing.
#[no_mangle]
pub extern "C" fn jl_gc_mmtk_dec_buf_flush() {
    // Intentionally empty.  All barrier buffers are now inside the
    // Mutator and flushed by Mutator::flush() during STW.
}

/// LXR write barrier for non-heap slots (e.g., external GenericMemory data).
///
/// Called BEFORE writing a pointer to a slot that is NOT in the MMTk managed
/// heap (e.g., malloc'd GenericMemory data buffers with `how != 0`).  The
/// caller provides the explicit old value (read from the slot before the
/// store) and the new value about to be stored.
///
/// This bypasses the generic barrier path (`enqueue_node`) entirely to avoid
/// two problems:
/// 1. **One-shot degradation (HANDOFF §53):** The old fallback to
///    `jl_gc_wb`/`object_probable_write` logged ALL fields of the parent,
///    causing subsequent writes to other in-heap fields to be silently missed.
/// 2. **RC overcount from slot re-reads:** Pushing the non-heap slot to the
///    incs queue causes `ProcessIncs` to re-read the slot at GC time.  For
///    repeated writes, the same final value is read N times → overcount by
///    N−1.  Using `JuliaVMSlot::Direct(new_val)` captures the value at
///    barrier time — each (old, new) pair produces exactly −1/+1 RC, with
///    no overcount.
///
/// The old value is pushed to `DEC_BUFFER` (same mechanism as
/// `jl_gc_mmtk_dec_buf_add`).  The new value is wrapped in
/// `JuliaVMSlot::Direct` and pushed to `NONHEAP_INC_BUFFER`, flushed as
/// `ProcessIncs` work packets at GC safepoints.  At GC time,
/// `Direct::load()` returns the captured value without any memory read;
/// `Direct::to_address()` returns `Address::ZERO` so `unlog_field_relaxed`,
/// `record_mature_evac_remset`, and `store` are all skipped.
#[no_mangle]
pub extern "C" fn mmtk_object_reference_write_pre_nonheap(
    mutator: *mut Mutator<JuliaVM>,
    src: ObjectReference,
    old_val: Address,
    new_val: Address,
) {
    // Genuinely non-heap slot write: the slot lives outside the MMTk heap, so
    // it has no field-unlog side metadata and the field-logging barrier cannot
    // operate on it.  This path is now reached only for the residual cases that
    // legitimately keep their data off-heap — e.g. `how == 2` foreign-wrapped
    // pointer arrays from `jl_ptr_to_genericmemory` (unsafe_wrap).
    //
    // Pointer-bearing GenericMemory buffers are NO LONGER off-heap: they are
    // allocated inline in the MMTk heap (LOS) so their slots carry side
    // metadata and take the regular in-heap barrier path
    // (`mmtk_object_reference_write_pre`).  See plan-alloc.md (Phase 1) and the
    // removed owner-rescan mechanism in mmtk-core (HANDOFF §2.13).
    //
    // For any remaining non-heap pointer slot, do the exact −1/+1: push the old
    // value to decrements and a `Direct(new)` slot to increments via the
    // mutator's own barrier queues (flushed by Mutator::flush() during STW).
    // `Direct` captures the value at barrier time, so ProcessIncs does not
    // re-read the (metadata-less) slot.
    let _ = src;
    let mutator = unsafe { &mut *mutator };
    let semantics = get_lxr_semantics(mutator);
    let old = if old_val.is_zero() {
        None
    } else {
        Some(unsafe { ObjectReference::from_raw_address_unchecked(old_val) })
    };
    if new_val.is_zero() {
        // No new value to increment; only decrement the old value.
        if let Some(old) = old {
            semantics.push_dec(old);
        }
    } else {
        let new_slot = crate::slots::JuliaVMSlot::Direct(unsafe {
            ObjectReference::from_raw_address_unchecked(new_val)
        });
        semantics.push_nonheap_write(old, new_slot);
    }
}

/// LXR write barrier pre-write (for field-logging barrier).
/// Logs the slot address and old value before a field store.
/// Under non-LXR plans, this is a no-op.
#[no_mangle]
pub extern "C" fn mmtk_object_reference_write_pre(
    mutator: *mut Mutator<JuliaVM>,
    src: ObjectReference,
    slot: Address,
    target: NullableObjectReference,
) {
    let mutator = unsafe { &mut *mutator };
    use mmtk::MutatorContext;
    mutator.barrier().object_reference_write_pre(
        src,
        crate::slots::JuliaVMSlot::Simple(mmtk::vm::slot::SimpleSlot::from_address(slot)),
        target.into(),
    );
}

/// LXR post-cmpswap write barrier with explicit old value.
///
/// Called AFTER a successful atomic compare-and-swap on a pointer field.
/// Unlike `mmtk_object_reference_write_pre`, this does NOT read the old
/// value from the slot (which now contains the new value).  Instead, the
/// caller provides the old value explicitly from the cmpswap's `expected`
/// parameter (unchanged on success per C11 semantics).
///
/// This is the correct barrier for cmpswap patterns under LXR.  A pre-write
/// barrier fired BEFORE a cmpswap that FAILS would spuriously decrement the
/// current value's RC on each retry iteration → premature free.
#[no_mangle]
pub extern "C" fn mmtk_object_reference_write_post_cmpswap(
    mutator: *mut Mutator<JuliaVM>,
    src: ObjectReference,
    slot: Address,
    old_val: Address,
    new_val: Address,
) {
    let mutator = unsafe { &mut *mutator };
    use mmtk::MutatorContext;
    mutator.barrier().object_reference_write_post_cmpswap(
        src,
        crate::slots::JuliaVMSlot::Simple(mmtk::vm::slot::SimpleSlot::from_address(slot)),
        ObjectReference::from_raw_address(old_val),
        ObjectReference::from_raw_address(new_val),
    );
}

/// LXR field-logging write barrier slow path.
/// This is the main barrier entry point for LXR. Under LXR's FieldBarrier,
/// object_reference_write_pre dispatches to LXRFieldBarrierSemantics::object_reference_write_slow
/// which performs all barrier operations in one pass:
/// 1. Logs old value for RC decrement
/// 2. Logs new value for RC increment
/// 3. Marks source for SATB concurrent tracing
/// 4. Updates remembered set for defragmentation
///
/// NOTE: FieldBarrier::object_reference_write_post is unimplemented!() — do NOT call it.
/// The pre-barrier handles everything for field-logging barriers.
#[no_mangle]
pub extern "C" fn mmtk_object_reference_write_field(
    mutator: *mut Mutator<JuliaVM>,
    src: ObjectReference,
    slot: Address,
    target: NullableObjectReference,
) {
    let mutator = unsafe { &mut *mutator };
    let slot = crate::slots::JuliaVMSlot::Simple(mmtk::vm::slot::SimpleSlot::from_address(slot));
    use mmtk::MutatorContext;
    // For FieldBarrier, object_reference_write_pre IS the full barrier.
    // It calls LXRFieldBarrierSemantics::object_reference_write_slow internally.
    mutator
        .barrier()
        .object_reference_write_pre(src, slot, target.into());
}

/// Query the active barrier type. Returns a string identifier.
/// This allows the Julia runtime to dispatch to the correct barrier calling convention.
#[no_mangle]
pub extern "C" fn mmtk_active_barrier() -> *const libc::c_char {
    use mmtk::plan::BarrierSelector;
    let barrier = SINGLETON.get_plan().constraints().barrier;
    match barrier {
        BarrierSelector::NoBarrier => b"NoBarrier\0".as_ptr() as *const libc::c_char,
        BarrierSelector::ObjectBarrier => b"ObjectBarrier\0".as_ptr() as *const libc::c_char,
        BarrierSelector::FieldBarrier => b"FieldBarrier\0".as_ptr() as *const libc::c_char,
        _ => b"Unknown\0".as_ptr() as *const libc::c_char,
    }
}

/// Export the base address of the field unlog bit side metadata.
/// Used by the Julia MLIR compiler to inline the barrier fast path.
#[no_mangle]
pub static MMTK_FIELD_UNLOG_BIT_BASE_ADDRESS: Address =
    mmtk::util::metadata::side_metadata::GLOBAL_SIDE_METADATA_VM_BASE_ADDRESS;
