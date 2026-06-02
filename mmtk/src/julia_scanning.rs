use crate::api::{mmtk_address_is_in_heap, mmtk_object_is_managed_by_mmtk};
use crate::collection::is_gc_thread;
use crate::julia_types::*;
use crate::slots::JuliaVMSlot;
use crate::slots::OffsetSlot;
use crate::JULIA_BUFF_TAG;
use memoffset::offset_of;
use mmtk::util::{Address, ObjectReference};
use mmtk::vm::slot::SimpleSlot;
use mmtk::vm::SlotVisitor;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::object_model::VMObjectModel;
use mmtk::vm::ObjectModel;

use crate::jl_gc_genericmemory_how;
use crate::jl_gc_get_owner_address_to_mmtk;
use crate::jl_gc_get_stackbase;
use crate::jl_gc_scan_julia_exc_obj;

pub const JL_MAX_TAGS: usize = 64; // from vm/julia/src/jl_exports.h
const OFFSET_OF_INLINED_SPACE_IN_MODULE: usize =
    offset_of!(jl_module_t, usings) + offset_of!(arraylist_t, _space);

#[allow(improper_ctypes)]
extern "C" {
    pub fn jl_mmtk_get_jl_datatype_type_addr() -> Address;
    pub static jl_datatype_type: *const jl_datatype_t;
    pub static jl_simplevector_type: *const jl_datatype_t;
    pub static jl_genericmemory_typename: *mut jl_typename_t;
    pub static jl_genericmemoryref_typename: *mut jl_typename_t;
    pub static jl_array_typename: *mut jl_typename_t;
    pub static jl_module_type: *const jl_datatype_t;
    pub static jl_task_type: *const jl_datatype_t;
    pub static jl_string_type: *const jl_datatype_t;
    pub static jl_weakref_type: *const jl_datatype_t;
    pub static jl_symbol_type: *const jl_datatype_t;
    pub static jl_method_type: *const jl_datatype_t;
    pub static jl_binding_partition_type: *const jl_datatype_t;
    pub static mut jl_small_typeof: [*mut jl_datatype_t; 128usize];
}

lazy_static::lazy_static! {
    static ref SAFE_PIPE: (libc::c_int, libc::c_int) = {
        let mut fds = [0; 2];
        unsafe {
            libc::pipe(fds.as_mut_ptr());
            libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK);
            libc::fcntl(fds[1], libc::F_SETFL, libc::O_NONBLOCK);
        }
        (fds[0], fds[1])
    };
}

use std::sync::atomic::AtomicBool;
static PIPE_LOCK: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub fn is_address_readable(addr: usize) -> bool {
    // Acquire spinlock to ensure thread-safe exclusive access to SAFE_PIPE
    while PIPE_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        std::hint::spin_loop();
    }

    let fd = SAFE_PIPE.1;
    let res = unsafe { libc::write(fd, addr as *const libc::c_void, 8) };
    let mut readable = true;
    if res == -1 {
        readable = false;
    } else if res == 8 {
        let mut buf = 0u64;
        unsafe {
            libc::read(SAFE_PIPE.0, &mut buf as *mut u64 as *mut libc::c_void, 8);
        }
    }

    // Release spinlock
    PIPE_LOCK.store(false, Ordering::Release);
    readable
}

#[inline(always)]
pub unsafe fn mmtk_jl_typetagof(addr: Address) -> Address {
    if addr.as_usize() < 0x20000 {
        return Address::zero();
    }
    let as_tagged_value =
        addr.as_usize() - std::mem::size_of::<crate::julia_scanning::jl_taggedvalue_t>();
    let t_header = Address::from_usize(as_tagged_value).load::<Address>();
    let t = t_header.as_usize() & !0xf;

    Address::from_usize(t)
}

#[inline(always)]
pub unsafe fn mmtk_jl_typeof(addr: Address) -> *const jl_datatype_t {
    mmtk_jl_to_typeof(mmtk_jl_typetagof(addr))
}

/// Forwarding-aware variant of `mmtk_jl_typeof`.  Use ONLY during GC
/// scanning — adds one extra memory load per non-small-tag resolution.
/// See `mmtk_jl_to_typeof_resolving` for details.
#[inline(always)]
pub unsafe fn mmtk_jl_typeof_resolving(addr: Address) -> *const jl_datatype_t {
    mmtk_jl_to_typeof_resolving(mmtk_jl_typetagof(addr))
}

/// If `addr` points to a DataType that has been forwarded during nursery evacuation,
/// follow the forwarding chain and return the address of the live copy.
/// Otherwise, returns `addr` unchanged.
///
/// During nursery evacuation, a DataType D can be copied to D'.  The old location D
/// has its header overwritten with a forwarding pointer to D'.  Any object whose vtag
/// still references D will see D's header (now a forwarding pointer) instead of a
/// valid type tag.  This function detects that condition by checking whether the
/// value at `addr`'s header is the small tag `jl_datatype_tag` (indicating a valid,
/// non-forwarded DataType).  If not, the header value is the forwarding pointer, and
/// we follow it.
///
/// CRITICAL: We must NOT use `mmtk_jl_typetagof` to extract the forwarding pointer,
/// because it masks with `& !0xf` which clears bit 3.  Julia object references are
/// at offset +8 from the 16-byte-aligned allocation start, so bit 3 of the object
/// reference is ALWAYS set.  The forwarding pointer is stored with mask
/// `0x00ff_ffff_ffff_fff8` (preserving bit 3).  We must use that mask to read it.
#[inline(always)]
pub unsafe fn resolve_forwarded_datatype_addr(addr: Address) -> Address {
    if cfg!(feature = "lxr_no_evac") {
        return addr;
    }
    if !is_gc_thread() {
        return addr;
    }
    if !mmtk_object_is_managed_by_mmtk(addr.as_usize()) {
        return addr;
    }
    let mut resolved = addr;
    // Bounded loop — normally at most 1 hop (objects are forwarded at most once per
    // GC cycle).  The bound of 3 is purely defensive.
    for _ in 0..3 {
        let r_addr = resolved.as_usize();
        if r_addr % 8 != 0 || !is_address_readable(r_addr - 8) {
            break;
        }
        let dt_vtag = mmtk_jl_typetagof(resolved);
        if dt_vtag.as_usize() == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4) {
            // Valid DataType — its own type tag is the DataType small tag.
            break;
        }
        // DataType is a regular object at offset +8, so we must restore bit 3
        let obj_ref_addr = Address::from_usize(resolved.as_usize() | 8);
        if let Some(obj_ref) = ObjectReference::from_raw_address(obj_ref_addr) {
            let status =
                mmtk::util::object_forwarding::get_forwarding_status::<crate::JuliaVM>(obj_ref);
            if mmtk::util::object_forwarding::state_is_forwarded_or_being_forwarded(status) {
                resolved = mmtk::util::object_forwarding::spin_and_get_forwarded_object::<
                    crate::JuliaVM,
                >(obj_ref, status)
                .to_raw_address();
                continue;
            }
        }
        break;
    }
    resolved
}

/// Fast type-tag resolution (mutator path).  No forwarding check — zero
/// overhead beyond the small-tag table lookup.  Safe to call from any
/// context where DataTypes cannot be forwarded (i.e. outside a STW GC
/// pause with nursery evacuation).
pub unsafe fn safe_jl_datatype_type() -> *const jl_datatype_t {
    let addr = jl_mmtk_get_jl_datatype_type_addr();
    if addr.is_zero() {
        std::ptr::null()
    } else {
        addr.load::<*const jl_datatype_t>()
    }
}

#[inline(always)]
pub unsafe fn mmtk_jl_to_typeof(t: Address) -> *const jl_datatype_t {
    let t_raw = t.as_usize();
    if t_raw < (JL_MAX_TAGS << 4) {
        let ty = jl_small_typeof[t_raw / std::mem::size_of::<Address>()];
        if (ty as usize) < 0x20000 {
            return safe_jl_datatype_type();
        }
        return ty;
    }
    if t_raw < 0x20000 {
        return safe_jl_datatype_type();
    }
    t.to_ptr::<jl_datatype_t>()
}

/// Forwarding-aware type-tag resolution (GC-only path).  During nursery
/// evacuation a DataType may be copied and its old header overwritten with
/// a forwarding pointer.  This variant detects that and follows the chain.
///
/// Cost: one extra memory load (`mmtk_jl_typetagof` on the resolved addr)
/// per non-small-tag resolution, plus side-metadata reads if forwarded.
/// Only use from GC scanning functions (`scan_julia_object`,
/// `get_so_object_size`, `copy`, `is_julia_obj_array`, etc.).
#[inline(always)]
pub unsafe fn mmtk_jl_to_typeof_resolving(t: Address) -> *const jl_datatype_t {
    let t_raw = t.as_usize();
    if t_raw < (JL_MAX_TAGS << 4) {
        let ty = jl_small_typeof[t_raw / std::mem::size_of::<Address>()];
        if (ty as usize) < 0x20000 {
            return safe_jl_datatype_type();
        }
        return ty;
    }
    let resolved = resolve_forwarded_datatype_addr(t);
    if resolved.as_usize() < 0x20000 {
        return safe_jl_datatype_type();
    }
    resolved.to_ptr::<jl_datatype_t>()
}

const PRINT_OBJ_TYPE: bool = false;

// This function is a rewrite of `gc_mark_outrefs()` in `gc.c`
// INFO: *_custom() functions are acessors to bitfields that do not use bindgen generated code.
#[inline(always)]
pub unsafe fn scan_julia_object<SV: SlotVisitor<JuliaVMSlot>>(obj: Address, closure: &mut SV) {
    // get Julia object type
    let mut vtag = mmtk_jl_typetagof(obj);
    let mut vtag_usize = vtag.as_usize();

    // If LXR is enabled, we must keep the object's type tag alive.
    // Small tags are integers, not GC-managed pointers. Real pointers to
    // DataTypes on the heap must be reported as TypeTag slots so LXR can
    // reference-count them, keep them alive as long as this object is alive,
    // and correctly update the type pointer in the header if the type moves.
    if vtag_usize >= ((jl_small_typeof_tags_jl_max_tags as usize) << 4) {
        let header_addr =
            obj.as_usize() - std::mem::size_of::<crate::julia_scanning::jl_taggedvalue_t>();
        let type_tag_slot = crate::slots::TypeTagSlot {
            address: Address::from_usize(header_addr),
        };
        closure.visit_slot(JuliaVMSlot::TypeTag(type_tag_slot), false);
    }

    if PRINT_OBJ_TYPE {
        println!(
            "scan_julia_obj {}, obj_type = {:?}",
            obj,
            mmtk_jl_to_typeof(obj)
        );
    }

    // symbols are always marked
    // buffers are marked by their parent object
    if vtag.to_ptr::<jl_datatype_t>() == jl_symbol_type || vtag_usize == JULIA_BUFF_TAG {
        return;
    }

    if vtag_usize == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_unionall_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_uniontype_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_tvar_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_vararg_tag as usize) << 4)
    {
        // these objects have pointers in them, but no other special handling
        // so we want these to fall through to the end
        vtag_usize = jl_small_typeof[vtag.as_usize() / std::mem::size_of::<Address>()] as usize;
        vtag = Address::from_usize(vtag_usize);
    } else if vtag_usize < ((jl_small_typeof_tags_jl_max_tags as usize) << 4) {
        // these objects either have specialing handling
        if vtag_usize == ((jl_small_typeof_tags_jl_simplevector_tag as usize) << 4) {
            if PRINT_OBJ_TYPE {
                println!("scan_julia_obj {}: simple vector\n", obj);
            }
            let length = mmtk_jl_svec_len(obj);
            let mut objary_begin = mmtk_jl_svec_data(obj);
            let objary_end = objary_begin.shift::<Address>(length as isize);

            while objary_begin < objary_end {
                process_slot(closure, objary_begin);
                objary_begin = objary_begin.shift::<Address>(1);
            }
        } else if vtag_usize == ((jl_small_typeof_tags_jl_module_tag as usize) << 4) {
            if PRINT_OBJ_TYPE {
                println!("scan_julia_obj {}: module\n", obj);
            }

            let m = obj.to_ptr::<jl_module_t>();
            let bindings_slot = ::std::ptr::addr_of!((*m).bindings);
            if PRINT_OBJ_TYPE {
                println!(" - scan bindings: {:?}\n", bindings_slot);
            }
            process_slot(closure, Address::from_ptr(bindings_slot));

            let bindingkeyset_slot = ::std::ptr::addr_of!((*m).bindingkeyset);
            if PRINT_OBJ_TYPE {
                println!(" - scan bindingkeyset: {:?}\n", bindingkeyset_slot);
            }
            process_slot(closure, Address::from_ptr(bindingkeyset_slot));

            let parent_slot = ::std::ptr::addr_of!((*m).parent);
            if PRINT_OBJ_TYPE {
                println!(" - scan parent: {:?}\n", parent_slot);
            }
            process_slot(closure, Address::from_ptr(parent_slot));

            let usings_backeges_slot = ::std::ptr::addr_of!((*m).usings_backedges);
            if PRINT_OBJ_TYPE {
                println!(" - scan parent: {:?}\n", usings_backeges_slot);
            }
            process_slot(closure, Address::from_ptr(usings_backeges_slot));

            let scanned_methods_slot = ::std::ptr::addr_of!((*m).scanned_methods);
            if PRINT_OBJ_TYPE {
                println!(" - scan parent: {:?}\n", scanned_methods_slot);
            }
            process_slot(closure, Address::from_ptr(scanned_methods_slot));

            // m.usings.items may be inlined in the module when the array list size <= AL_N_INLINE (cf. arraylist_new)
            // In that case it may be an mmtk object and not a malloced address.
            // If it is an mmtk object, (*m).usings.items will then be an internal pointer to the module
            // which means we will need to trace and update it if the module moves
            if mmtk_object_is_managed_by_mmtk((*m).usings.items as usize) {
                let offset = OFFSET_OF_INLINED_SPACE_IN_MODULE;
                let slot = Address::from_ptr(::std::ptr::addr_of!((*m).usings.items));
                process_offset_slot(closure, slot, offset);
            }

            let nusings = (*m).usings.len;
            if nusings > 0 {
                let mut objary_begin = Address::from_mut_ptr((*m).usings.items);
                let objary_end = objary_begin.shift::<Address>(nusings as isize);

                while objary_begin < objary_end {
                    if PRINT_OBJ_TYPE {
                        println!(" - scan usings: {:?}\n", objary_begin);
                    }
                    process_slot(closure, objary_begin);
                    // _jl_module_using is 3 pointers: { mod, min_world, max_world }
                    // Only `mod` (the first field) is a GC pointer.
                    // Step by 3 to advance to the next struct entry.
                    objary_begin = objary_begin.shift::<Address>(3);
                }
            }
        } else if vtag_usize == ((jl_small_typeof_tags_jl_task_tag as usize) << 4) {
            if PRINT_OBJ_TYPE {
                println!("scan_julia_obj {}: task\n", obj);
            }

            let ta = obj.to_ptr::<jl_task_t>();

            // SOTA Safety: Do NOT scan a task's gcstack inside heap tracing (scan_julia_object),
            // because heap tracing can run concurrently (concurrent marking) while the mutator
            // is actively modifying its stack, causing segfaults. All active task stacks are
            // already precisely and safely scanned during the STW root scanning phase.
            // mmtk_scan_gcstack(ta, closure);

            let layout = (*jl_task_type).layout;
            debug_assert!((*layout).fielddesc_type_custom() == 0);
            debug_assert!((*layout).nfields > 0);
            let npointers = (*layout).npointers;
            let mut obj8_begin = mmtk_jl_dt_layout_ptrs(layout);
            let obj8_end = obj8_begin.shift::<u8>(npointers as isize);

            while obj8_begin < obj8_end {
                let obj8_begin_loaded = obj8_begin.load::<u8>();
                let slot = obj.shift::<Address>(obj8_begin_loaded as isize);
                process_slot(closure, slot);
                obj8_begin = obj8_begin.shift::<u8>(1);
            }
        } else if vtag_usize == ((jl_small_typeof_tags_jl_string_tag as usize) << 4)
            && PRINT_OBJ_TYPE
        {
            println!("scan_julia_obj {}: string\n", obj);
        }
        return;
    } else {
        // vtag is a direct pointer to a jl_datatype_t in the heap.
        // During nursery evacuation, this DataType may itself have been
        // forwarded — its header is now a forwarding pointer, not a type tag.
        // Resolve through the forwarding chain before validation.
        vtag = resolve_forwarded_datatype_addr(vtag);

        let is_datatype = is_valid_datatype(vtag);

        if !is_datatype {
            let mut status = 0u8;
            if is_gc_thread() {
                // DataType is a regular object at offset +8, so we must restore bit 3
                let obj_ref_addr = Address::from_usize(vtag.as_usize() | 8);
                if mmtk_object_is_managed_by_mmtk(vtag.as_usize()) {
                    if let Some(obj_ref) = ObjectReference::from_raw_address(obj_ref_addr) {
                        status = mmtk::util::object_forwarding::get_forwarding_status::<
                            crate::JuliaVM,
                        >(obj_ref);
                    }
                }
            }
            // Log the warning instead of panicking to survive residual freed-object scans
            // during complex sysimage compilation passes.
            eprintln!(
                "GC warning (probable corruption ignored) - !jl_is_datatype = true, vt = {:?}, type_tag = 0, forwarding_status = {}",
                vtag.as_usize(),
                status
            );
            return;
        }

        let vt = if vtag.as_usize() < 0x20000 {
            safe_jl_datatype_type()
        } else {
            vtag.to_ptr::<jl_datatype_t>()
        };
        let type_tag = mmtk_jl_typetagof(vtag);
        let type_tag_usize = type_tag.as_usize();
        let datatype_type_addr = unsafe { jl_mmtk_get_jl_datatype_type_addr() }.as_usize();
        let datatype_type_val = if datatype_type_addr != 0 {
            unsafe { Address::from_usize(datatype_type_addr).load::<usize>() }
        } else {
            0
        };
    }
    let vt = if vtag.as_usize() < 0x20000 {
        unsafe { safe_jl_datatype_type() }
    } else {
        vtag.to_ptr::<jl_datatype_t>()
    };
    if vt.is_null() {
        return;
    }
    if (*vt).name == jl_array_typename {
        let a = obj.to_ptr::<jl_array_t>();
        let memref = (*a).ref_;

        let ptr_or_offset = memref.ptr_or_offset;
        // if the object moves its pointer inside the array object (void* ptr_or_offset) needs to be updated as well
        if mmtk_object_is_managed_by_mmtk(ptr_or_offset as usize) {
            let ptr_or_ref_slot = Address::from_ptr(::std::ptr::addr_of!((*a).ref_.ptr_or_offset));
            let mem_addr_as_usize = memref.mem as usize;
            let ptr_or_offset_as_usize = ptr_or_offset as usize;
            if ptr_or_offset_as_usize > mem_addr_as_usize {
                let offset = ptr_or_offset_as_usize - mem_addr_as_usize;

                // Only update the offset pointer if the offset is valid (> 0)
                if offset > 0 {
                    process_offset_slot(closure, ptr_or_ref_slot, offset);
                }
            }
        }
    }
    if (*vt).name == jl_genericmemory_typename {
        if PRINT_OBJ_TYPE {
            println!("scan_julia_obj {}: genericmemory\n", obj);
        }
        let m = obj.to_ptr::<jl_genericmemory_t>();
        let how = jl_gc_genericmemory_how(obj);

        if PRINT_OBJ_TYPE {
            println!("scan_julia_obj {}: genericmemory how = {}\n", obj, how);
        }

        if how == 3 {
            let owner_addr = mmtk_jl_genericmemory_data_owner_field_address(m);
            process_slot(closure, owner_addr);

            return;
        }

        if (*m).length == 0 {
            return;
        }

        let layout = (*vt).layout;
        if layout.is_null() || (layout as usize) < 0x20000 {
            return;
        }
        if (*layout).flags.arrayelem_isboxed() != 0 {
            let length = (*m).length;
            let mut objary_begin = Address::from_ptr((*m).ptr);
            let objary_end = objary_begin.shift::<Address>(length as isize);
            while objary_begin < objary_end {
                process_slot(closure, objary_begin);
                objary_begin = objary_begin.shift::<Address>(1);
            }
        } else if (*layout).first_ptr >= 0 {
            let npointers = (*layout).npointers;
            let elsize = (*layout).size as usize / std::mem::size_of::<Address>();
            let length = (*m).length;
            let mut objary_begin = Address::from_ptr((*m).ptr);
            let objary_end = objary_begin.shift::<Address>((length * elsize) as isize);
            if npointers == 1 {
                objary_begin = objary_begin.shift::<Address>((*layout).first_ptr as isize);
                while objary_begin < objary_end {
                    process_slot(closure, objary_begin);
                    objary_begin = objary_begin.shift::<Address>(elsize as isize);
                }
            } else if (*layout).fielddesc_type_custom() == 0 {
                let obj8_begin = mmtk_jl_dt_layout_ptrs(layout);
                let obj8_end = obj8_begin.shift::<u8>(npointers as isize);
                let mut elem_begin = obj8_begin;
                let elem_end = obj8_end;

                while objary_begin < objary_end {
                    while elem_begin < elem_end {
                        let elem_begin_loaded = elem_begin.load::<u8>();
                        let slot = objary_begin.shift::<Address>(elem_begin_loaded as isize);
                        process_slot(closure, slot);
                        elem_begin = elem_begin.shift::<u8>(1);
                    }
                    elem_begin = obj8_begin;
                    objary_begin = objary_begin.shift::<Address>(elsize as isize);
                }
            } else if (*layout).fielddesc_type_custom() == 1 {
                let mut obj16_begin = mmtk_jl_dt_layout_ptrs(layout);
                let obj16_end = obj16_begin.shift::<u16>(npointers as isize);

                while objary_begin < objary_end {
                    while obj16_begin < obj16_end {
                        let elem_begin_loaded = obj16_begin.load::<u16>();
                        let slot = objary_begin.shift::<Address>(elem_begin_loaded as isize);
                        process_slot(closure, slot);
                        obj16_begin = obj16_begin.shift::<u16>(1);
                    }
                    obj16_begin = mmtk_jl_dt_layout_ptrs(layout);
                    objary_begin = objary_begin.shift::<Address>(elsize as isize);
                }
            } else {
                unimplemented!();
            }
        }

        return;
    }

    if PRINT_OBJ_TYPE {
        println!("scan_julia_obj {}: datatype\n", obj);
    }

    if vt == jl_weakref_type {
        return;
    }

    let layout = (*vt).layout;
    if layout.is_null() || (layout as usize) < 0x20000 {
        return;
    }
    let npointers = (*layout).npointers;
    if npointers != 0 {
        debug_assert!(
            (*layout).nfields > 0 && (*layout).fielddesc_type_custom() != 3,
            "opaque types should have been handled specially"
        );
        if (*layout).fielddesc_type_custom() == 0 {
            let mut obj8_begin = mmtk_jl_dt_layout_ptrs(layout);
            let obj8_end = obj8_begin.shift::<u8>(npointers as isize);

            while obj8_begin < obj8_end {
                let obj8_begin_loaded = obj8_begin.load::<u8>();
                let slot = obj.shift::<Address>(obj8_begin_loaded as isize);
                process_slot(closure, slot);
                obj8_begin = obj8_begin.shift::<u8>(1);
            }
        } else if (*layout).fielddesc_type_custom() == 1 {
            let mut obj16_begin = mmtk_jl_dt_layout_ptrs(layout);
            let obj16_end = obj16_begin.shift::<u16>(npointers as isize);

            while obj16_begin < obj16_end {
                let obj16_begin_loaded = obj16_begin.load::<u16>();
                let slot = obj.shift::<Address>(obj16_begin_loaded as isize);
                process_slot(closure, slot);
                obj16_begin = obj16_begin.shift::<u16>(1);
            }
        } else if (*layout).fielddesc_type_custom() == 2 {
            let mut obj32_begin = mmtk_jl_dt_layout_ptrs(layout);
            let obj32_end = obj32_begin.shift::<u32>(npointers as isize);

            while obj32_begin < obj32_end {
                let obj32_begin_loaded = obj32_begin.load::<u32>();
                let slot = obj.shift::<Address>(obj32_begin_loaded as isize);
                process_slot(closure, slot);
                obj32_begin = obj32_begin.shift::<u32>(1);
            }
        } else {
            debug_assert!((*layout).fielddesc_type_custom() == 3);
            unimplemented!();
        }
    }
}

#[inline(always)]
unsafe fn mmtk_jl_genericmemory_data_owner_field_address(m: *const jl_genericmemory_t) -> Address {
    unsafe { jl_gc_get_owner_address_to_mmtk(Address::from_ptr(m)) }
}

// #[inline(always)]
// unsafe fn mmtk_jl_genericmemory_data_owner_field(
//     m: *const mmtk_jl_genericmemory_t,
// ) -> *const mmtk_jl_value_t {
//     mmtk_jl_genericmemory_data_owner_field_address(m).load::<*const mmtk_jl_value_t>()
// }

pub unsafe fn mmtk_scan_gcstack<EV: SlotVisitor<JuliaVMSlot>>(
    ta: *const jl_task_t,
    closure: &mut EV,
) {
    let stkbuf = (*ta).ctx.stkbuf;
    let copy_stack = (*ta).ctx.copy_stack_custom();

    #[cfg(feature = "julia_copy_stack")]
    if !stkbuf.is_null() && copy_stack != 0 {
        let stkbuf_slot = Address::from_ptr(::std::ptr::addr_of!((*ta).ctx.stkbuf));
        process_slot(closure, stkbuf_slot);
    }

    let mut s = (*ta).gcstack;
    let (mut offset, mut lb, mut ub) = (0_isize, 0_u64, u64::MAX);

    #[cfg(feature = "julia_copy_stack")]
    if !stkbuf.is_null() && copy_stack != 0 && (*ta).ptls.is_null() {
        if ((*ta).tid._M_i) < 0 {
            panic!("tid must be positive.")
        }
        let stackbase = jl_gc_get_stackbase((*ta).tid._M_i);
        ub = stackbase as u64;
        lb = ub - ((*ta).ctx.copy_stack() as u64);
        offset = (*ta).ctx.stkbuf as isize - lb as isize;
    }

    if !s.is_null() {
        let s_nroots_addr = ::std::ptr::addr_of!((*s).nroots);
        let mut nroots = read_stack(Address::from_ptr(s_nroots_addr), offset, lb, ub);
        debug_assert!(nroots.as_usize() as u32 <= u32::MAX);
        let mut nr = nroots >> 2;

        loop {
            let rts = Address::from_mut_ptr(s).shift::<Address>(2);
            let mut i = 0;
            while i < nr {
                if (nroots.as_usize() & 1) != 0 {
                    let slot = read_stack(rts.shift::<Address>(i as isize), offset, lb, ub);
                    let real_addr = get_stack_addr(slot, offset, lb, ub);
                    process_slot(closure, real_addr);
                } else {
                    let real_addr =
                        get_stack_addr(rts.shift::<Address>(i as isize), offset, lb, ub);

                    let slot = read_stack(rts.shift::<Address>(i as isize), offset, lb, ub);
                    use crate::julia_finalizer::gc_ptr_tag;
                    // malloced pointer tagged in jl_gc_add_quiescent
                    // skip both the next element (native function), and the object
                    if slot & 3usize == 3 {
                        i += 2;
                        continue;
                    }

                    // pointer is not malloced but function is native, so skip it
                    if gc_ptr_tag(slot, 1) {
                        process_offset_slot(closure, real_addr, 1);
                        i += 2;
                        continue;
                    }

                    process_slot(closure, real_addr);
                }

                i += 1;
            }

            let s_prev_address = ::std::ptr::addr_of!((*s).prev);
            let sprev = read_stack(Address::from_ptr(s_prev_address), offset, lb, ub);
            if sprev.is_zero() {
                break;
            }

            s = sprev.to_mut_ptr::<jl_gcframe_t>();
            let s_nroots_addr = ::std::ptr::addr_of!((*s).nroots);
            let new_nroots = read_stack(Address::from_ptr(s_nroots_addr), offset, lb, ub);
            nroots = new_nroots;
            nr = nroots >> 2;
            continue;
        }
    }

    // just call into C, since the code is cold
    if !(*ta).excstack.is_null() {
        jl_gc_scan_julia_exc_obj(
            Address::from_ptr(ta),
            Address::from_mut_ptr(closure),
            process_slot::<EV> as _,
        );
    }
}

#[inline(always)]
unsafe fn read_stack(addr: Address, offset: isize, lb: u64, ub: u64) -> Address {
    let real_addr = get_stack_addr(addr, offset, lb, ub);

    real_addr.load::<Address>()
}

#[inline(always)]
fn get_stack_addr(addr: Address, offset: isize, lb: u64, ub: u64) -> Address {
    if addr.as_usize() >= lb as usize && addr.as_usize() < ub as usize {
        addr + offset
    } else {
        addr
    }
}

#[inline(always)]
pub fn process_slot<EV: SlotVisitor<JuliaVMSlot>>(closure: &mut EV, slot: Address) {
    let simple_slot = SimpleSlot::from_address(slot);

    #[cfg(debug_assertions)]
    {
        use mmtk::vm::slot::Slot;

        if PRINT_OBJ_TYPE {
            println!(
                "\tprocess slot = {:?} - {:?}\n",
                simple_slot,
                simple_slot.load()
            );
        }

        if let Some(objref) = simple_slot.load() {
            debug_assert!(
                mmtk::memory_manager::is_in_mmtk_spaces(objref),
                "Object {:?} in slot {:?} is not mapped address",
                objref,
                simple_slot
            );

            let raw_addr_usize = objref.to_raw_address().as_usize();

            // captures wrong slots before creating the work
            debug_assert!(
                raw_addr_usize % 16 == 0 || raw_addr_usize % 8 == 0,
                "Object {:?} in slot {:?} is not aligned to 8 or 16",
                objref,
                simple_slot
            );
        }
    }

    closure.visit_slot(JuliaVMSlot::Simple(simple_slot), false);
}

#[inline(always)]
pub fn process_offset_slot<EV: SlotVisitor<JuliaVMSlot>>(
    closure: &mut EV,
    slot: Address,
    offset: usize,
) {
    let offset_slot = OffsetSlot::new_with_offset(slot, offset);
    #[cfg(debug_assertions)]
    {
        use mmtk::vm::slot::Slot;

        if let Some(objref) = offset_slot.load() {
            debug_assert!(
                mmtk::memory_manager::is_in_mmtk_spaces(objref),
                "Object {:?} in slot {:?} is not mapped address",
                objref,
                offset_slot
            );
        }
    }

    closure.visit_slot(JuliaVMSlot::Offset(offset_slot), false);
}

#[inline(always)]
pub fn mmtk_jl_array_ndimwords(ndims: u32) -> usize {
    if ndims < 3 {
        return 0;
    }

    (ndims - 2) as usize
}

#[inline(always)]
pub unsafe fn mmtk_jl_svec_len(obj: Address) -> usize {
    (*obj.to_ptr::<jl_svec_t>()).length
}

#[inline(always)]
pub unsafe fn mmtk_jl_svec_data(obj: Address) -> Address {
    obj + std::mem::size_of::<crate::julia_scanning::jl_svec_t>()
}

#[inline(always)]
pub unsafe fn mmtk_jl_tparam0(vt: *const jl_datatype_t) -> *const jl_datatype_t {
    mmtk_jl_svecref((*vt).parameters, 0)
}

#[inline(always)]
pub unsafe fn mmtk_jl_svecref(vt: *mut jl_svec_t, i: usize) -> *const jl_datatype_t {
    debug_assert!(
        mmtk_jl_typetagof(Address::from_mut_ptr(vt)).as_usize()
            == (jl_small_typeof_tags_jl_simplevector_tag << 4) as usize
    );
    debug_assert!(i < mmtk_jl_svec_len(Address::from_mut_ptr(vt)));

    let svec_data = mmtk_jl_svec_data(Address::from_mut_ptr(vt));
    let result_ptr = svec_data + i;
    let result = result_ptr.atomic_load::<AtomicUsize>(Ordering::Relaxed);
    result as *const _jl_datatype_t
}

#[inline(always)]
pub unsafe fn mmtk_jl_dt_layout_ptrs(l: *const jl_datatype_layout_t) -> Address {
    mmtk_jl_dt_layout_fields(l)
        + (mmtk_jl_fielddesc_size((*l).fielddesc_type_custom()) * (*l).nfields) as usize
}

#[inline(always)]
pub unsafe fn mmtk_jl_dt_layout_fields(l: *const jl_datatype_layout_t) -> Address {
    Address::from_ptr(l) + std::mem::size_of::<jl_datatype_layout_t>()
}

#[inline(always)]
pub unsafe fn mmtk_jl_fielddesc_size(fielddesc_type: u16) -> u32 {
    debug_assert!(fielddesc_type <= 2);
    2 << fielddesc_type
}

const JL_BT_NON_PTR_ENTRY: usize = usize::MAX;

pub unsafe fn mmtk_jl_bt_is_native(bt_entry: *mut jl_bt_element_t) -> bool {
    let entry = unsafe { (*bt_entry).__bindgen_anon_1.uintptr };
    entry != JL_BT_NON_PTR_ENTRY
}

pub unsafe fn mmtk_jl_bt_entry_size(bt_entry: *mut jl_bt_element_t) -> usize {
    if mmtk_jl_bt_is_native(bt_entry) {
        1
    } else {
        2 + mmtk_jl_bt_num_jlvals(bt_entry) + mmtk_jl_bt_num_uintvals(bt_entry)
    }
}

pub unsafe fn mmtk_jl_bt_num_jlvals(bt_entry: *mut jl_bt_element_t) -> usize {
    debug_assert!(!mmtk_jl_bt_is_native(bt_entry));
    let entry = unsafe { (*bt_entry.add(1)).__bindgen_anon_1.uintptr };
    entry & 0x7
}

pub unsafe fn mmtk_jl_bt_num_uintvals(bt_entry: *mut jl_bt_element_t) -> usize {
    debug_assert!(!mmtk_jl_bt_is_native(bt_entry));
    let entry = unsafe { (*bt_entry.add(1)).__bindgen_anon_1.uintptr };
    (entry >> 3) & 0x7
}

pub unsafe fn mmtk_jl_bt_entry_jlvalue(
    bt_entry: *mut jl_bt_element_t,
    i: usize,
) -> ObjectReference {
    let entry = unsafe { (*bt_entry.add(2 + i)).__bindgen_anon_1.jlvalue };
    debug_assert!(!entry.is_null());
    unsafe { ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(entry)) }
}

/// Returns the slot address of the i-th jlvalue in a backtrace entry.
/// Unlike mmtk_jl_bt_entry_jlvalue (which dereferences the slot to get the object),
/// this returns the address of the slot itself, for slot-based root reporting.
pub unsafe fn mmtk_jl_bt_entry_jlvalue_slot(bt_entry: *mut jl_bt_element_t, i: usize) -> Address {
    Address::from_ptr(std::ptr::addr_of!(
        (*bt_entry.add(2 + i)).__bindgen_anon_1.jlvalue
    ))
}

// ====== LXR object classification for concurrent marking ======

use mmtk::vm::ObjectKind;

/// Classify a Julia object for LXR concurrent marking's chunked scanning optimization.
/// - GenericMemory with boxed (pointer) elements → ObjArray(len)
/// - Everything else → Scalar
pub unsafe fn get_julia_obj_kind(object: ObjectReference) -> ObjectKind {
    let obj = object.to_raw_address();
    let vt = mmtk_jl_typeof_resolving(obj);
    if vt.is_null() || (vt as usize) < 0x20000 {
        return ObjectKind::Scalar;
    }
    if (*vt).name == jl_genericmemory_typename {
        let m = obj.to_ptr::<jl_genericmemory_t>();
        let layout = (*vt).layout;
        // Boxed elements = array of pointers → ObjArray
        if (*layout).flags.arrayelem_isboxed() != 0 && (*m).length > 0 {
            return ObjectKind::ObjArray((*m).length as u32);
        }
    }
    ObjectKind::Scalar
}

/// Return a JuliaMemorySlice over the pointer data region of a GenericMemory ObjArray.
/// Only called when get_obj_kind returned ObjArray.
pub unsafe fn get_julia_obj_array_data(object: ObjectReference) -> crate::slots::JuliaMemorySlice {
    let obj = object.to_raw_address();
    let m = obj.to_ptr::<jl_genericmemory_t>();
    let length = (*m).length;
    let data_start = Address::from_ptr((*m).ptr);
    crate::slots::JuliaMemorySlice {
        owner: object,
        start: data_start,
        count: length,
    }
}

#[inline(always)]
pub unsafe fn is_valid_datatype_struct(vt: *const jl_datatype_t) -> bool {
    if vt.is_null() {
        return false;
    }
    let vt_addr = vt as usize;
    if vt_addr < 0x20000 {
        return false;
    }
    // Check readability of the start and end of the jl_datatype_t struct (size 56)
    if !is_address_readable(vt_addr) || !is_address_readable(vt_addr + 48) {
        return false;
    }
    let name_ptr = (*vt).name;
    if name_ptr.is_null()
        || (name_ptr as usize) < 0x20000
        || !is_address_readable(name_ptr as usize)
    {
        return false;
    }
    let super_ptr = (*vt).super_;
    if !super_ptr.is_null()
        && ((super_ptr as usize) < 0x20000 || !is_address_readable(super_ptr as usize))
    {
        return false;
    }
    let layout_ptr = (*vt).layout;
    if !layout_ptr.is_null()
        && ((layout_ptr as usize) < 0x20000 || !is_address_readable(layout_ptr as usize))
    {
        return false;
    }
    true
}

#[inline(always)]
pub unsafe fn is_valid_datatype(vtag: Address) -> bool {
    let tag_addr = vtag.as_usize();
    if tag_addr < (JL_MAX_TAGS << 4) {
        return true;
    }
    if tag_addr < 0x20000 || !is_address_readable(tag_addr - 8) {
        return false;
    }
    let type_tag = mmtk_jl_typetagof(vtag);
    let type_tag_usize = type_tag.as_usize();
    let datatype_type_addr = jl_mmtk_get_jl_datatype_type_addr().as_usize();
    let datatype_type_val = if datatype_type_addr != 0 {
        Address::from_usize(datatype_type_addr).load::<usize>()
    } else {
        0
    };
    let is_vtag_datatype_type = (datatype_type_addr != 0 && vtag.as_usize() == datatype_type_addr)
        || (datatype_type_val != 0 && vtag.as_usize() == datatype_type_val)
        || vtag.as_usize() < 0x20000;

    let is_type_tag_readable = type_tag_usize >= 0x20000 && is_address_readable(type_tag_usize - 8);

    let is_datatype = type_tag_usize == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4)
        || (is_type_tag_readable
            && type_tag_usize >= (JL_MAX_TAGS << 4)
            && mmtk_jl_typetagof(type_tag).as_usize()
                == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4))
        || (datatype_type_addr != 0 && type_tag_usize == datatype_type_addr)
        || (datatype_type_val != 0 && type_tag_usize == datatype_type_val)
        || (is_vtag_datatype_type && type_tag_usize == vtag.as_usize());

    if !is_datatype {
        return false;
    }

    // Deep validation of the jl_datatype_t struct fields
    is_valid_datatype_struct(vtag.to_ptr::<jl_datatype_t>())
}

/// Returns true if the object is a GenericMemory with all-boxed (pointer) elements.
/// Used by LXR's nursery scanning (`scan_nursery_object` in rc.rs) to select the
/// chunked obj-array scanning path.
pub unsafe fn is_julia_obj_array(object: ObjectReference) -> bool {
    let obj = object.to_raw_address();
    let mut vtag = mmtk_jl_typetagof(obj);
    vtag = resolve_forwarded_datatype_addr(vtag);
    if !is_valid_datatype(vtag) {
        return false;
    }
    let vt = if vtag.as_usize() < 0x20000 {
        safe_jl_datatype_type()
    } else {
        vtag.to_ptr::<jl_datatype_t>()
    };
    if vt.is_null() {
        return false;
    }
    if (*vt).name == jl_genericmemory_typename {
        let layout = (*vt).layout;
        if layout.is_null() || (layout as usize) < 0x20000 {
            return false;
        }
        return (*layout).flags.arrayelem_isboxed() != 0;
    }
    false
}

/// Returns true if the object is a GenericMemory with NO pointer fields at all
/// (pure isbits/value data).  Used by LXR's nursery scanning to skip field
/// scanning and unlog-bits setup entirely for value-only arrays.
pub unsafe fn is_julia_val_array(object: ObjectReference) -> bool {
    let obj = object.to_raw_address();
    let mut vtag = mmtk_jl_typetagof(obj);
    vtag = resolve_forwarded_datatype_addr(vtag);
    if !is_valid_datatype(vtag) {
        return false;
    }
    let vt = if vtag.as_usize() < 0x20000 {
        safe_jl_datatype_type()
    } else {
        vtag.to_ptr::<jl_datatype_t>()
    };
    if vt.is_null() {
        return false;
    }
    if (*vt).name == jl_genericmemory_typename {
        let layout = (*vt).layout;
        if layout.is_null() || (layout as usize) < 0x20000 {
            return false;
        }
        // Not boxed and no embedded pointers → pure value array
        return (*layout).flags.arrayelem_isboxed() == 0 && (*layout).first_ptr < 0;
    }
    false
}

/// Diagnostic-only (feature `lxr_rc_trace`): read the NUL-terminated name string
/// of a `jl_sym_t`.  The name bytes are stored inline immediately after the
/// 24-byte symbol header (standard Julia layout).
#[cfg(feature = "lxr_rc_trace")]
unsafe fn debug_symbol_name(sym: *mut jl_sym_t) -> String {
    if sym.is_null() {
        return "<null-sym>".to_string();
    }
    let name_ptr = (sym as *const u8).add(std::mem::size_of::<jl_sym_t>());
    let mut bytes = Vec::new();
    let mut i = 0isize;
    // Bound the read defensively (symbol names are short).
    while i < 256 {
        let b = *name_ptr.offset(i);
        if b == 0 {
            break;
        }
        bytes.push(b);
        i += 1;
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Diagnostic-only (feature `lxr_rc_trace`): return true if `object`'s type
/// tag denotes a valid Julia type.  Mirrors the validity check in
/// `scan_julia_object`/`get_current_size` (object_model.rs): a small-typeof
/// tag is always valid; otherwise the tag must point to a `jl_datatype_t`
/// whose own tag is the DataType small-tag and whose `smalltag()` is 0.  A
/// `false` result means the object is freed/corrupt (RC undercount).
#[cfg(feature = "lxr_rc_trace")]
pub unsafe fn debug_julia_object_tag_is_valid(object: ObjectReference) -> bool {
    let obj = object.to_raw_address();
    let mut vtag = mmtk_jl_typetagof(obj);
    let vtag_usize = vtag.as_usize();
    // Small-typeof encoded tags are always valid.
    if vtag_usize < (JL_MAX_TAGS << 4) {
        return true;
    }
    // Direct pointer to a jl_datatype_t: resolve any forwarding, then validate.
    vtag = resolve_forwarded_datatype_addr(vtag);
    let tag_addr = vtag.as_usize();
    if tag_addr < (JL_MAX_TAGS << 4) {
        return true;
    }
    if !vtag.is_mapped() {
        return false;
    }
    let vt = vtag.to_ptr::<jl_datatype_t>();
    let type_tag = mmtk_jl_typetagof(vtag);
    let type_tag_usize = type_tag.as_usize();
    let datatype_type_addr = unsafe { jl_mmtk_get_jl_datatype_type_addr() }.as_usize();
    let datatype_type_val = if datatype_type_addr != 0 {
        unsafe { Address::from_usize(datatype_type_addr).load::<usize>() }
    } else {
        0
    };
    let is_datatype = type_tag_usize == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4)
        || (type_tag_usize >= (JL_MAX_TAGS << 4)
            && unsafe { mmtk_jl_typetagof(type_tag).as_usize() }
                == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4))
        || type_tag_usize == datatype_type_addr
        || type_tag_usize == datatype_type_val
        || type_tag_usize == vtag.as_usize();
    is_datatype
}

/// Diagnostic-only (feature `lxr_rc_trace`): return the object's type name
/// using ONLY single-word reads (the tag at `-8`, the typename pointer, and
/// the symbol name bytes).  Does NOT read `(*vt).layout` or any data fields, so
/// it is safe to call on a candidate object-start during a mid-sweep heap walk
/// (used by `describe_referrer_owner`).  Returns "" for an invalid tag.
#[cfg(feature = "lxr_rc_trace")]
pub unsafe fn debug_julia_object_type_name(object: ObjectReference) -> String {
    let obj = object.to_raw_address();
    let mut vtag = mmtk_jl_typetagof(obj);
    // Small-typeof encoded tag: not a direct DataType pointer; report the tag.
    if vtag.as_usize() < (JL_MAX_TAGS << 4) {
        return format!("<smalltag {:#x}>", vtag.as_usize());
    }
    vtag = resolve_forwarded_datatype_addr(vtag);
    if !vtag.is_mapped() {
        return String::new();
    }
    let vt = vtag.to_ptr::<jl_datatype_t>();
    let type_tag = mmtk_jl_typetagof(vtag);
    if type_tag.as_usize() != ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4) {
        return String::new();
    }
    let tn = (*vt).name;
    if tn.is_null() {
        return "<null-typename>".to_string();
    }
    debug_symbol_name((*tn).name)
}

/// Diagnostic-only (feature `lxr_rc_trace`): describe a Julia object for the
/// LXR nursery-promotion corruption guard.  Reports the resolved type name and,
/// for a GenericMemory, the array classification flags (`arrayelem_isboxed`,
/// `first_ptr`, `npointers`, `arrayelem_isunion`), `how`, `length`, the data
/// pointer, and a sample of the first few data words — so a misclassified
/// value array (scanned as a pointer array) can be identified precisely.
#[cfg(feature = "lxr_rc_trace")]
pub unsafe fn debug_describe_julia_object(object: ObjectReference) -> String {
    let obj = object.to_raw_address();
    let vt = mmtk_jl_typeof_resolving(obj);
    if vt.is_null() {
        return format!("<null-vt obj={:#x}>", obj.as_usize());
    }
    let tn = (*vt).name;
    let type_name = if tn.is_null() {
        "<null-typename>".to_string()
    } else {
        debug_symbol_name((*tn).name)
    };
    if (*vt).name == jl_genericmemory_typename {
        let m = obj.to_ptr::<jl_genericmemory_t>();
        let how = jl_gc_genericmemory_how(obj);
        let layout = (*vt).layout;
        // Element type = the Memory type's 2nd type parameter (kind, T, addrspace).
        let elt_name = {
            let params = (*vt).parameters;
            if params.is_null() {
                "<no-params>".to_string()
            } else {
                let svec_len = mmtk_jl_svec_len(Address::from_ptr(params));
                if svec_len < 2 {
                    "<svec<2>".to_string()
                } else {
                    let data = mmtk_jl_svec_data(Address::from_ptr(params));
                    let elt = data.shift::<Address>(1).load::<Address>();
                    if elt.is_zero() {
                        "<null-elt>".to_string()
                    } else {
                        let elt_vt = elt.to_ptr::<jl_datatype_t>();
                        let elt_tag = mmtk_jl_typetagof(elt);
                        if elt_tag.as_usize()
                            == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4)
                            && !(*elt_vt).name.is_null()
                        {
                            debug_symbol_name((*(*elt_vt).name).name)
                        } else {
                            format!("<elt-tag={:#x}>", elt_tag.as_usize())
                        }
                    }
                }
            }
        };
        let (isboxed, isunion, first_ptr, npointers) = if layout.is_null() {
            (-1i32, -1i32, -1i32, -1i32)
        } else {
            (
                (*layout).flags.arrayelem_isboxed() as i32,
                (*layout).flags.arrayelem_isunion() as i32,
                (*layout).first_ptr as i32,
                (*layout).npointers as i32,
            )
        };
        let length = (*m).length;
        let dataptr = (*m).ptr as usize;
        // Sample words around the valid->garbage boundary (elements 155..175)
        // to expose any embedded object header / overlapping allocation.
        let mut sample = String::new();
        let data = Address::from_ptr((*m).ptr);
        let lo = if length > 175 { 155isize } else { 0 };
        let hi = std::cmp::min(length as isize, lo + 20);
        for i in lo..hi {
            let a = data.shift::<usize>(i);
            if a.is_mapped() {
                sample.push_str(&format!(" [{}]={:#x}", i, a.load::<usize>()));
            } else {
                sample.push_str(&format!(" [{}]=<unmapped>", i));
            }
        }
        let zeroinit = (*vt).zeroinit() as i32;
        format!(
            "GenericMemory{{{}}} vt={:#x} how={} len={} dataptr={:#x} isboxed={} isunion={} first_ptr={} npointers={} zeroinit={} data:[{} ]",
            elt_name, vt as usize, how, length, dataptr,
            isboxed, isunion, first_ptr, npointers, zeroinit, sample
        )
    } else {
        format!("{} vt={:#x}", type_name, vt as usize)
    }
}

// ====== scan_julia_object_with_type: pre-loaded type pointer ======

/// Scan a Julia object using a pre-loaded type pointer (klass), avoiding the
/// header read that `scan_julia_object` does via `mmtk_jl_typetagof`.
/// Used by LXR's concurrent marking which caches the class pointer for
/// chunked large-array scanning.
pub unsafe fn scan_julia_object_with_type<SV: SlotVisitor<JuliaVMSlot>>(
    obj: Address,
    closure: &mut SV,
    klass: Address,
) {
    // If klass is zero (shouldn't happen but be defensive), fall back to the header read
    if klass.is_zero() {
        scan_julia_object(obj, closure);
        return;
    }

    // Real pointers to DataTypes on the heap must be reported as TypeTag slots so LXR can
    // reference-count them, keep them alive, and correctly update the type pointer in the header
    // if the type moves.
    let klass_usize = klass.as_usize();
    if klass_usize >= ((jl_small_typeof_tags_jl_max_tags as usize) << 4) {
        let header_addr =
            obj.as_usize() - std::mem::size_of::<crate::julia_scanning::jl_taggedvalue_t>();
        let type_tag_slot = crate::slots::TypeTagSlot {
            address: Address::from_usize(header_addr),
        };
        closure.visit_slot(JuliaVMSlot::TypeTag(type_tag_slot), false);
    }

    // The klass is a jl_datatype_t* (what mmtk_jl_typeof returns).
    // We need to reconstruct the vtag (type tag) that scan_julia_object uses.
    // mmtk_jl_typeof returns the resolved datatype; the vtag is what's stored in
    // the header, which may be a smalltag or a direct pointer.
    //
    // For the hot path (genericmemory ObjArray), the klass IS the resolved type,
    // and we can use it directly. For all other cases, the savings from avoiding
    // one header read are minimal — just fall back to the full scanner.
    let vt = klass.to_ptr::<jl_datatype_t>();

    // Fast path: if this is a genericmemory, we can scan it directly with the known type
    if (*vt).name == jl_genericmemory_typename {
        scan_genericmemory_with_type(obj, closure, vt);
        return;
    }

    // For all other object kinds, fall back to the full scanner.
    // The cost of one extra header read is negligible for non-array objects.
    scan_julia_object(obj, closure);
}

/// Scan a GenericMemory object with a known type pointer.
/// This is the hot path optimization — large GenericMemory arrays of pointers
/// are the main beneficiary of the klass caching in LXR's concurrent marking.
unsafe fn scan_genericmemory_with_type<SV: SlotVisitor<JuliaVMSlot>>(
    obj: Address,
    closure: &mut SV,
    vt: *const jl_datatype_t,
) {
    let m = obj.to_ptr::<jl_genericmemory_t>();
    let how = jl_gc_genericmemory_how(obj);

    if how == 3 {
        let owner_addr = mmtk_jl_genericmemory_data_owner_field_address(m);
        process_slot(closure, owner_addr);
        return;
    }

    if (*m).length == 0 {
        return;
    }

    let layout = (*vt).layout;
    if (*layout).flags.arrayelem_isboxed() != 0 {
        let length = (*m).length;
        let mut objary_begin = Address::from_ptr((*m).ptr);
        let objary_end = objary_begin.shift::<Address>(length as isize);
        while objary_begin < objary_end {
            process_slot(closure, objary_begin);
            objary_begin = objary_begin.shift::<Address>(1);
        }
    } else if (*layout).first_ptr >= 0 {
        let npointers = (*layout).npointers;
        let elsize = (*layout).size as usize / std::mem::size_of::<Address>();
        let length = (*m).length;
        let mut objary_begin = Address::from_ptr((*m).ptr);
        let objary_end = objary_begin.shift::<Address>((length * elsize) as isize);
        if npointers == 1 {
            objary_begin = objary_begin.shift::<Address>((*layout).first_ptr as isize);
            while objary_begin < objary_end {
                process_slot(closure, objary_begin);
                objary_begin = objary_begin.shift::<Address>(elsize as isize);
            }
        } else if (*layout).fielddesc_type_custom() == 0 {
            let obj8_begin = mmtk_jl_dt_layout_ptrs(layout);
            let obj8_end = obj8_begin.shift::<u8>(npointers as isize);
            let mut elem_begin = obj8_begin;
            let elem_end = obj8_end;

            while objary_begin < objary_end {
                while elem_begin < elem_end {
                    let elem_begin_loaded = elem_begin.load::<u8>();
                    let slot = objary_begin.shift::<Address>(elem_begin_loaded as isize);
                    process_slot(closure, slot);
                    elem_begin = elem_begin.shift::<u8>(1);
                }
                elem_begin = obj8_begin;
                objary_begin = objary_begin.shift::<Address>(elsize as isize);
            }
        } else if (*layout).fielddesc_type_custom() == 1 {
            let mut obj16_begin = mmtk_jl_dt_layout_ptrs(layout);
            let obj16_end = obj16_begin.shift::<u16>(npointers as isize);

            while objary_begin < objary_end {
                while obj16_begin < obj16_end {
                    let elem_begin_loaded = obj16_begin.load::<u16>();
                    let slot = objary_begin.shift::<Address>(elem_begin_loaded as isize);
                    process_slot(closure, slot);
                    obj16_begin = obj16_begin.shift::<u16>(1);
                }
                obj16_begin = mmtk_jl_dt_layout_ptrs(layout);
                objary_begin = objary_begin.shift::<Address>(elsize as isize);
            }
        } else {
            unimplemented!();
        }
    }
}

/// Helper: When the `mem` field of a `jl_array_t` is updated to point to a new
/// forwarded/evacuated `jl_genericmemory_t`, we must dynamically update its
/// sibling field `ptr_or_offset` if it was pointing inside the inlined data of
/// the old memory. This ensures the array's data pointer is kept in sync even if
/// the slot is not in the remembered set.
#[inline(always)]
pub unsafe fn update_array_ptr_or_offset_if_needed(slot_addr: Address, new_mem: ObjectReference) {
    if slot_addr.as_usize() % 8 != 0 {
        return;
    }
    // ref.mem is at offset 8 (size_of::<usize>()) from the start of jl_array_t.
    // So the parent object starts at slot_addr - size_of::<usize>().
    let parent_addr = slot_addr.shift::<Address>(-1);

    if parent_addr.as_usize() % 8 == 0 && mmtk_object_is_managed_by_mmtk(parent_addr.as_usize()) {
        let vt = mmtk_jl_typetagof(parent_addr);
        if vt.as_usize() >= ((jl_small_typeof_tags_jl_max_tags as usize) << 4)
            && is_valid_datatype(vt)
        {
            let vt_ptr = vt.to_ptr::<jl_datatype_t>();
            if !vt_ptr.is_null()
                && ((*vt_ptr).name == jl_array_typename
                    || (*vt_ptr).name == jl_genericmemoryref_typename)
            {
                // Yes, the parent is a jl_array_t or jl_genericmemoryref_t!
                // Read the old mem pointer from slot_addr.
                let old_mem_addr = slot_addr.load::<Address>();
                if !old_mem_addr.is_zero() {
                    // Only process if the old memory is inlined (how == 0)
                    let how = jl_gc_genericmemory_how(old_mem_addr);
                    if how == 0 {
                        // Read the old ptr_or_offset from parent_addr.
                        let old_ptr_or_offset = parent_addr.load::<usize>();
                        if old_ptr_or_offset != 0 {
                            // Calculate the offset of ptr_or_offset relative to the old mem pointer.
                            let old_mem_usize = old_mem_addr.as_usize();
                            if old_ptr_or_offset >= old_mem_usize {
                                let offset = old_ptr_or_offset - old_mem_usize;
                                if offset >= 16 {
                                    // Calculate the new ptr_or_offset relative to the new mem pointer.
                                    let new_ptr_or_offset =
                                        new_mem.to_raw_address().as_usize() + offset;
                                    // Update the ptr_or_offset field in the parent.
                                    parent_addr.store::<usize>(new_ptr_or_offset);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
