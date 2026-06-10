#![allow(clippy::missing_safety_doc)]

#[cfg(feature = "hosted")]
extern crate libc;
extern crate log;
extern crate mmtk;
#[cfg(feature = "hosted")]
#[macro_use]
extern crate lazy_static;

#[cfg(feature = "hosted")]
use mmtk::util::opaque_pointer::*;
#[cfg(feature = "hosted")]
use mmtk::util::Address;
#[cfg(feature = "hosted")]
use mmtk::vm::VMBinding;
#[cfg(feature = "hosted")]
use mmtk::MMTKBuilder;
#[cfg(feature = "hosted")]
use mmtk::MMTK;

#[cfg(feature = "hosted")]
use std::collections::HashMap;
#[cfg(feature = "hosted")]
use std::sync::atomic::AtomicIsize;
#[cfg(feature = "hosted")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "hosted")]
use std::sync::{Arc, Condvar, Mutex, RwLock};

#[cfg(feature = "hosted")]
pub mod active_plan;
#[cfg(feature = "hosted")]
pub mod api;
#[cfg(feature = "hosted")]
mod build_info;
#[cfg(feature = "hosted")]
pub mod collection;
#[cfg(feature = "hosted")]
pub mod gc_trigger;
#[cfg(feature = "hosted")]
pub mod object_model;
#[cfg(feature = "hosted")]
pub mod reference_glue;
#[cfg(feature = "hosted")]
pub mod scanning;
#[cfg(feature = "hosted")]
pub mod slots;
#[cfg(feature = "hosted")]
pub mod util;

#[cfg(feature = "hosted")]
pub mod julia_finalizer;
#[cfg(feature = "hosted")]
pub mod julia_scanning;
#[cfg(feature = "hosted")]
#[allow(non_camel_case_types)]
#[allow(improper_ctypes_definitions)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
#[allow(clippy::all)]
#[rustfmt::skip]
pub mod julia_types;

/// VM binding for julia-mlir compiled binaries.
/// Activated by `--features julia_mlir`. Mutually exclusive with the
/// default Julia-hosted binding.
#[cfg(not(feature = "hosted"))]
pub mod julia_mlir;

#[cfg(feature = "hosted")]
#[derive(Default)]
pub struct JuliaVM;

#[cfg(feature = "hosted")]
use crate::slots::JuliaVMSlot;

#[cfg(feature = "hosted")]
impl VMBinding for JuliaVM {
    const MAX_ALIGNMENT: usize = 64;
    const MIN_ALIGNMENT: usize = 4;
    type VMObjectModel = object_model::VMObjectModel;
    type VMScanning = scanning::VMScanning;
    type VMCollection = collection::VMCollection;
    type VMActivePlan = active_plan::VMActivePlan;
    type VMReferenceGlue = reference_glue::VMReferenceGlue;
    type VMMemorySlice = slots::JuliaMemorySlice;
    type VMSlot = JuliaVMSlot;
}

#[cfg(feature = "hosted")]
/// This is used to ensure we initialize MMTk at a specified timing.
pub static MMTK_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "hosted")]
lazy_static! {
    pub static ref BUILDER: Mutex<MMTKBuilder> = Mutex::new(MMTKBuilder::new());
    pub static ref SINGLETON: MMTK<JuliaVM> = {
        let builder = BUILDER.lock().unwrap();
        debug_assert!(!MMTK_INITIALIZED.load(Ordering::SeqCst));
        let ret = mmtk::memory_manager::mmtk_init(&builder);
        MMTK_INITIALIZED.store(true, std::sync::atomic::Ordering::Relaxed);
        *ret
    };
}

#[cfg(feature = "hosted")]
pub static mut JULIA_HEADER_SIZE: usize = 0;
#[cfg(feature = "hosted")]
pub static mut JULIA_BUFF_TAG: usize = 0;

#[cfg(feature = "hosted")]
#[no_mangle]
pub static BLOCK_FOR_GC: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "hosted")]
#[no_mangle]
pub static WORLD_HAS_STOPPED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "hosted")]
#[no_mangle]
pub static DISABLED_GC: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "hosted")]
#[no_mangle]
pub static USER_TRIGGERED_GC: AtomicIsize = AtomicIsize::new(0);

#[cfg(feature = "hosted")]
lazy_static! {
    pub static ref STW_COND: Arc<(Mutex<usize>, Condvar)> =
        Arc::new((Mutex::new(0), Condvar::new()));
    pub static ref STOP_MUTATORS: Arc<(Mutex<usize>, Condvar)> =
        Arc::new((Mutex::new(0), Condvar::new()));

    // We create a boxed mutator with MMTk core, and we mem copy its content to jl_tls_state_t (shallow copy).
    // This map stores the pair of the mutator address in jl_tls_state_t and the original boxed mutator.
    // As we only do a shallow copy, we should not free the original boxed mutator, until the thread is getting destroyed.
    // Otherwise, we will have dangling pointers.
    pub static ref MUTATORS: RwLock<HashMap<Address, Address>> = RwLock::new(HashMap::new());
}

#[cfg(feature = "hosted")]
type ProcessSlotFn = *const extern "C" fn(closure: Address, slot: Address);

#[cfg(feature = "hosted")]
#[allow(improper_ctypes)]
extern "C" {
    pub fn jl_gc_scan_julia_exc_obj(obj: Address, closure: Address, process_slot: ProcessSlotFn);
    pub fn jl_gc_get_stackbase(tid: i16) -> usize;
    pub fn jl_throw_out_of_memory_error();
    pub fn jl_get_gc_disable_counter() -> u32;
    pub fn jl_gc_mmtk_sweep_malloced_memory();
    pub fn jl_gc_sweep_stack_pools_and_mtarraylist_buffers();
    pub fn jl_hrtime() -> u64;
    pub fn jl_gc_update_stats(t: u64, mmtk_live_bytes: usize, is_nursery: bool);
    pub fn jl_gc_get_abi_structs_checksum_c() -> usize;
    pub fn jl_gc_get_thread_finalizer_list(tls: OpaquePointer) -> Address;
    pub fn jl_gc_get_to_finalize_list() -> Address;
    pub fn jl_gc_get_marked_finalizers_list() -> Address;
    pub fn arraylist_grow(a: Address, n: usize);
    pub fn jl_gc_get_have_pending_finalizers() -> *mut i32;
    pub fn jl_gc_scan_vm_specific_roots(closure: *mut crate::slots::RootsWorkClosure);
    pub fn jl_gc_update_inlined_array(to: Address, from: Address);
    pub fn jl_gc_prepare_to_collect();
    pub fn jl_gc_get_owner_address_to_mmtk(m: Address) -> Address;
    pub fn jl_gc_genericmemory_how(m: Address) -> usize;
    pub fn jl_gc_get_max_memory() -> usize;
    pub static mut MMTK_SIDE_LOG_BIT_BASE_ADDRESS: Address;
}

#[cfg(feature = "hosted")]
pub(crate) fn set_panic_hook() {
    let old_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |panic_info| {
        if crate::collection::is_gc_thread() {
            eprintln!("ERROR: An MMTk GC thread panicked.  This is a bug.");
            eprintln!("{panic_info}");

            let bt = std::backtrace::Backtrace::capture();
            match bt.status() {
                std::backtrace::BacktraceStatus::Unsupported => {
                    eprintln!("Backtrace is unsupported.")
                }
                std::backtrace::BacktraceStatus::Disabled => {
                    eprintln!("Backtrace is disabled.");
                    eprintln!(
                        "run with `RUST_BACKTRACE=1` environment variable to display a backtrace"
                    );
                }
                std::backtrace::BacktraceStatus::Captured => {
                    eprintln!("{bt}");
                }
                s => {
                    eprintln!("Unknown backtrace status: {s:?}");
                }
            }

            std::process::abort();
        } else {
            old_hook(panic_info);
        }
    }));
}
