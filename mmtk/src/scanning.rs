use crate::slots::JuliaVMSlot;
use crate::SINGLETON;
use mmtk::memory_manager;
use mmtk::scheduler::*;
use mmtk::util::opaque_pointer::*;
use mmtk::util::ObjectReference;
use mmtk::vm::slot::SimpleSlot;
use mmtk::vm::slot::Slot;
use mmtk::vm::ObjectKind;
use mmtk::vm::ObjectTracerContext;
use mmtk::vm::RootsWorkFactory;
use mmtk::vm::Scanning;
use mmtk::vm::SlotVisitor;
use mmtk::vm::VMBinding;
use mmtk::Mutator;
use mmtk::MMTK;

use crate::jl_gc_mmtk_sweep_malloced_memory;
use crate::jl_gc_scan_vm_specific_roots;
use crate::jl_gc_sweep_stack_pools_and_mtarraylist_buffers;
use crate::JuliaVM;

pub struct VMScanning {}

/// Collects JuliaVMSlot entries from the gcstack scanner.
/// Unlike the old SlotBuffer (which extracted ObjectReferences from slots),
/// this preserves the actual slot addresses so LXR can process them
/// through its slot-based RC increment path (ProcessIncs).
struct GCStackSlotBuffer {
    pub buffer: Vec<JuliaVMSlot>,
}

impl mmtk::vm::SlotVisitor<JuliaVMSlot> for GCStackSlotBuffer {
    fn visit_slot(&mut self, slot: JuliaVMSlot, _out_of_heap: bool) {
        // Only include slots that actually point to something
        if slot.load().is_some() {
            self.buffer.push(slot);
        }
    }
}

/// Scan a single ptls struct, collecting root slots (not objects) into the buffers.
///
/// Root slots are the C-heap field addresses (e.g., &ptls->root_task) — these are
/// reported as JuliaVMSlot::Simple so LXR's RC increment path can process them.
/// GC stack slots are collected via GCStackSlotBuffer which preserves slot addresses.
unsafe fn scan_ptls_roots(
    ptls: &mut crate::julia_types::_jl_tls_states_t,
    root_slots: &mut Vec<JuliaVMSlot>,
    gcstack_slots: &mut GCStackSlotBuffer,
) {
    use crate::julia_scanning::*;
    use crate::julia_types::*;
    use mmtk::util::Address;

    /// Helper: scan a task's gcstack and optionally add the task's slot as a root.
    /// `slot_addr` is the address of the C-heap field holding the task pointer.
    unsafe fn scan_task_at_slot(
        slot_addr: Address,
        task: *const _jl_task_t,
        task_is_root: bool,
        gcstack_slots: &mut GCStackSlotBuffer,
        root_slots: &mut Vec<JuliaVMSlot>,
    ) {
        if !task.is_null() {
            // Scan the task's gcstack — produces actual slot addresses
            mmtk_scan_gcstack(task, gcstack_slots);

            if task_is_root {
                // Report the C-heap slot holding the task pointer
                // (e.g., &ptls->root_task) so LXR can RC-increment the task object
                root_slots.push(JuliaVMSlot::Simple(SimpleSlot::from_address(slot_addr)));
            }
        }
    }

    // Root task  (*mut _jl_task_t)
    scan_task_at_slot(
        Address::from_ptr(std::ptr::addr_of!(ptls.root_task)),
        ptls.root_task as *const _jl_task_t,
        true,
        gcstack_slots,
        root_slots,
    );

    // Live tasks (not roots themselves — only scan their gcstacks)
    let mut i = 0;
    while i < ptls.gc_tls_common.heap.live_tasks.len {
        let mut task_address = Address::from_ptr(ptls.gc_tls_common.heap.live_tasks.items);
        task_address = task_address.shift::<Address>(i as isize);
        let task = task_address.load::<*const jl_task_t>();
        if !task.is_null() {
            mmtk_scan_gcstack(task, gcstack_slots);
        }
        i += 1;
    }

    // Current task  (u64 — raw address stored as integer)
    {
        let current_task = ptls.current_task as *const _jl_task_t;
        scan_task_at_slot(
            Address::from_ptr(std::ptr::addr_of!(ptls.current_task)),
            current_task,
            true,
            gcstack_slots,
            root_slots,
        );
    }

    // Next task  (*mut _jl_task_t)
    scan_task_at_slot(
        Address::from_ptr(std::ptr::addr_of!(ptls.next_task)),
        ptls.next_task as *const _jl_task_t,
        true,
        gcstack_slots,
        root_slots,
    );

    // Previous task  (*mut _jl_task_t)
    scan_task_at_slot(
        Address::from_ptr(std::ptr::addr_of!(ptls.previous_task)),
        ptls.previous_task as *const _jl_task_t,
        true,
        gcstack_slots,
        root_slots,
    );

    // Previous exception  (*mut jl_value_t)
    if !ptls.previous_exception.is_null() {
        let slot_addr = Address::from_ptr(std::ptr::addr_of!(ptls.previous_exception));
        root_slots.push(JuliaVMSlot::Simple(SimpleSlot::from_address(slot_addr)));
    }

    // Backtrace buffer jlvalues — these are slot addresses in the bt_data array
    let mut i = 0;
    while i < ptls.bt_size {
        let bt_entry = ptls.bt_data.add(i);
        let bt_entry_size = mmtk_jl_bt_entry_size(bt_entry);
        if mmtk_jl_bt_is_native(bt_entry) {
            i += bt_entry_size;
            continue;
        }
        let njlvals = mmtk_jl_bt_num_jlvals(bt_entry);
        for j in 0..njlvals {
            // Get the address of the bt entry slot itself
            let bt_slot_addr = mmtk_jl_bt_entry_jlvalue_slot(bt_entry, j);
            root_slots.push(JuliaVMSlot::Simple(SimpleSlot::from_address(bt_slot_addr)));
        }
        i += bt_entry_size;
    }
}

impl Scanning<JuliaVM> for VMScanning {
    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        mutator: &'static mut Mutator<JuliaVM>,
        mut factory: impl RootsWorkFactory<JuliaVMSlot>,
    ) {
        // NOTE: All barrier inc/dec buffers now live inside the Mutator's
        // LXRFieldBarrierSemantics (not in thread_local! storage).  They are
        // flushed by Mutator::flush() which is called on EVERY mutator thread
        // in StopMutators::do_work — no manual flush is needed anywhere.
        // See HANDOFF Pitfall #58 for history.

        let ptls: &mut crate::julia_types::_jl_tls_states_t =
            unsafe { std::mem::transmute(mutator.mutator_tls) };

        let mut gcstack_slots = GCStackSlotBuffer { buffer: vec![] };
        let mut root_slots: Vec<JuliaVMSlot> = vec![];

        unsafe {
            scan_ptls_roots(ptls, &mut root_slots, &mut gcstack_slots);
        }

        // Report all roots as slot-based work.
        // LXR processes slots through RCImmixCollectRootEdges -> ProcessIncs (RC increment path).
        // ProcessIncs guards non-Immix/LOS objects, so sysimage/immortal objects are safe.
        const CAPACITY_PER_PACKET: usize = 4096;

        // Combine gcstack slots and root slots into a single stream
        // (they all go through the same RC increment path)
        let all_slots: Vec<JuliaVMSlot> = gcstack_slots
            .buffer
            .into_iter()
            .chain(root_slots.into_iter())
            .collect();

        for chunk in all_slots.chunks(CAPACITY_PER_PACKET).map(|c| c.to_vec()) {
            factory.create_process_roots_work(chunk, mmtk::scheduler::RootKind::Strong);
        }
    }

    fn scan_multiple_thread_root(
        _tls: VMWorkerThread,
        mutators: Vec<VMMutatorThread>,
        mut factory: impl RootsWorkFactory<JuliaVMSlot>,
    ) {
        // NOTE: All barrier inc/dec buffers are inside the Mutator and
        // flushed by Mutator::flush() during STW.  No manual flush needed.
        // See comment in scan_roots_in_mutator_thread above.

        let mut gcstack_slots = GCStackSlotBuffer { buffer: vec![] };
        let mut root_slots: Vec<JuliaVMSlot> = vec![];

        for mutator_tls in mutators {
            let ptls: &mut crate::julia_types::_jl_tls_states_t =
                unsafe { std::mem::transmute(mutator_tls) };

            unsafe {
                scan_ptls_roots(ptls, &mut root_slots, &mut gcstack_slots);
            }
        }

        const CAPACITY_PER_PACKET: usize = 4096;

        let all_slots: Vec<JuliaVMSlot> = gcstack_slots
            .buffer
            .into_iter()
            .chain(root_slots.into_iter())
            .collect();

        for chunk in all_slots.chunks(CAPACITY_PER_PACKET).map(|c| c.to_vec()) {
            factory.create_process_roots_work(chunk, mmtk::scheduler::RootKind::Strong);
        }
    }

    fn scan_vm_specific_roots(
        _tls: VMWorkerThread,
        mut factory: impl RootsWorkFactory<JuliaVMSlot>,
    ) {
        use crate::slots::RootsWorkClosure;
        let mut roots_closure = RootsWorkClosure::from_roots_work_factory(&mut factory);
        unsafe {
            jl_gc_scan_vm_specific_roots(&mut roots_closure as _);
        }
    }

    fn scan_object(
        _tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut impl SlotVisitor<JuliaVMSlot>,
    ) {
        process_object(object, slot_visitor);
    }
    fn scan_object_with_klass(
        _tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut impl SlotVisitor<JuliaVMSlot>,
        klass: mmtk::util::Address,
    ) {
        // Use the pre-loaded type pointer (klass) to avoid re-reading the header
        let addr = object.to_raw_address();
        unsafe {
            crate::julia_scanning::scan_julia_object_with_type(addr, slot_visitor, klass);
        }
    }
    fn get_obj_kind(object: ObjectReference) -> ObjectKind {
        unsafe { crate::julia_scanning::get_julia_obj_kind(object) }
    }

    fn is_obj_array(object: ObjectReference) -> bool {
        unsafe { crate::julia_scanning::is_julia_obj_array(object) }
    }

    fn is_val_array(object: ObjectReference) -> bool {
        unsafe { crate::julia_scanning::is_julia_val_array(object) }
    }

    fn obj_array_data(object: ObjectReference) -> crate::slots::JuliaMemorySlice {
        unsafe { crate::julia_scanning::get_julia_obj_array_data(object) }
    }

    #[cfg(feature = "lxr_rc_trace")]
    fn debug_describe_object(object: ObjectReference) -> String {
        unsafe { crate::julia_scanning::debug_describe_julia_object(object) }
    }

    #[cfg(feature = "lxr_rc_trace")]
    fn debug_object_tag_is_valid(object: ObjectReference) -> bool {
        unsafe { crate::julia_scanning::debug_julia_object_tag_is_valid(object) }
    }

    #[cfg(feature = "lxr_rc_trace")]
    fn debug_object_type_name(object: ObjectReference) -> String {
        unsafe { crate::julia_scanning::debug_julia_object_type_name(object) }
    }

    fn notify_initial_thread_scan_complete(_partial_scan: bool, _tls: VMWorkerThread) {
        let sweep_vm_specific_work = SweepVMSpecific::new();
        memory_manager::add_work_packet(
            &SINGLETON,
            WorkBucketStage::Compact,
            sweep_vm_specific_work,
        );
    }
    fn supports_return_barrier() -> bool {
        unimplemented!()
    }

    fn prepare_for_roots_re_scanning() {
        unimplemented!()
    }

    fn process_weak_refs(
        _worker: &mut GCWorker<JuliaVM>,
        tracer_context: impl ObjectTracerContext<JuliaVM>,
    ) -> bool {
        let single_thread_process_finalizer = ScanFinalizersSingleThreaded { tracer_context };
        memory_manager::add_work_packet(
            &SINGLETON,
            WorkBucketStage::VMRefClosure,
            single_thread_process_finalizer,
        );

        // We have pushed work. No need to repeat this method.
        false
    }
}

pub fn process_object(object: ObjectReference, closure: &mut impl SlotVisitor<JuliaVMSlot>) {
    let addr = object.to_raw_address();
    unsafe {
        crate::julia_scanning::scan_julia_object(addr, closure);
    }
}

// Sweep malloced arrays work
pub struct SweepVMSpecific {
    swept: bool,
}

impl SweepVMSpecific {
    pub fn new() -> Self {
        Self { swept: false }
    }
}

impl Default for SweepVMSpecific {
    fn default() -> Self {
        Self::new()
    }
}

impl<VM: VMBinding> GCWork<VM> for SweepVMSpecific {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        // call sweep malloced arrays and sweep stack pools
        unsafe { jl_gc_mmtk_sweep_malloced_memory() }
        unsafe { jl_gc_sweep_stack_pools_and_mtarraylist_buffers() }
        self.swept = true;
    }
}

pub struct ScanFinalizersSingleThreaded<C: ObjectTracerContext<JuliaVM>> {
    tracer_context: C,
}

impl<C: ObjectTracerContext<JuliaVM>> GCWork<JuliaVM> for ScanFinalizersSingleThreaded<C> {
    fn do_work(&mut self, worker: &mut GCWorker<JuliaVM>, _mmtk: &'static MMTK<JuliaVM>) {
        self.tracer_context.with_tracer(worker, |tracer| {
            crate::julia_finalizer::scan_finalizers_in_rust(tracer);
        });
    }
}
