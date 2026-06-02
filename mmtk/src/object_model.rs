use crate::api::mmtk_get_obj_size;
use crate::jl_gc_genericmemory_how;
use crate::jl_gc_update_inlined_array;
use crate::julia_scanning::{
    jl_genericmemory_typename, jl_small_typeof, mmtk_jl_typeof, mmtk_jl_typeof_resolving,
    mmtk_jl_typetagof, resolve_forwarded_datatype_addr,
};
use crate::julia_types::*;
use crate::{JuliaVM, JULIA_BUFF_TAG, JULIA_HEADER_SIZE};
use log::trace;
use mmtk::util::copy::*;
use mmtk::util::{Address, ObjectReference};
use mmtk::vm::ObjectModel;
use mmtk::vm::*;

pub struct VMObjectModel {}

/// Global logging bit metadata spec
/// 1 bit per object
pub(crate) const LOGGING_SIDE_METADATA_SPEC: VMGlobalLogBitSpec = VMGlobalLogBitSpec::side_first();

/// Global field-level unlog bit metadata spec (for LXR field-logging barrier)
/// 1 bit per pointer-sized slot
pub(crate) const FIELD_LOGGING_SIDE_METADATA_SPEC: VMGlobalFieldUnlogBitSpec =
    VMGlobalFieldUnlogBitSpec::side_first();

pub(crate) const MARKING_METADATA_SPEC: VMLocalMarkBitSpec =
    VMLocalMarkBitSpec::side_after(LOS_METADATA_SPEC.as_spec());

pub(crate) const LOCAL_PINNING_METADATA_BITS_SPEC: VMLocalPinningBitSpec =
    VMLocalPinningBitSpec::side_after(MARKING_METADATA_SPEC.as_spec());

// pub(crate) const LOCAL_FORWARDING_POINTER_METADATA_SPEC: VMLocalForwardingPointerSpec =
//     VMLocalForwardingPointerSpec::side_after(MARKING_METADATA_SPEC.as_spec());

// pub(crate) const LOCAL_FORWARDING_METADATA_BITS_SPEC: VMLocalForwardingBitsSpec =
//     VMLocalForwardingBitsSpec::side_after(LOCAL_FORWARDING_POINTER_METADATA_SPEC.as_spec());

/// PolicySpecific mark-and-nursery bits metadata spec
/// 2-bits per object
pub(crate) const LOS_METADATA_SPEC: VMLocalLOSMarkNurserySpec =
    VMLocalLOSMarkNurserySpec::side_first();

impl ObjectModel<JuliaVM> for VMObjectModel {
    const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec = LOGGING_SIDE_METADATA_SPEC;
    const GLOBAL_FIELD_UNLOG_BIT_SPEC: VMGlobalFieldUnlogBitSpec = FIELD_LOGGING_SIDE_METADATA_SPEC;
    const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec =
        VMLocalForwardingPointerSpec::in_header(-64);

    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec =
        VMLocalForwardingBitsSpec::side_after(LOCAL_PINNING_METADATA_BITS_SPEC.as_spec());

    const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec = MARKING_METADATA_SPEC;
    const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec = LOS_METADATA_SPEC;
    const UNIFIED_OBJECT_REFERENCE_ADDRESS: bool = false;
    const OBJECT_REF_OFFSET_LOWER_BOUND: isize = 0;

    const LOCAL_PINNING_BIT_SPEC: VMLocalPinningBitSpec = LOCAL_PINNING_METADATA_BITS_SPEC;

    fn try_copy(
        from: ObjectReference,
        semantics: CopySemantics,
        copy_context: &mut GCWorkerCopyContext<JuliaVM>,
    ) -> Option<ObjectReference> {
        // Delegate to the infallible copy() implementation.
        // Julia's copy always succeeds (alloc_copy never returns zero).
        Some(Self::copy(from, semantics, copy_context))
    }

    fn copy(
        from: ObjectReference,
        semantics: CopySemantics,
        copy_context: &mut GCWorkerCopyContext<JuliaVM>,
    ) -> ObjectReference {
        trace!("Attempting to copy object {}", from);

        let bytes = Self::get_current_size(from);
        let from_addr = from.to_raw_address();
        let from_start = Self::ref_to_object_start(from);
        let header_offset = from_addr - from_start;

        let dst = if header_offset == 8 {
            // regular object
            // Note: The `from` reference is not used by any allocator currently in MMTk core.
            copy_context.alloc_copy(from, bytes, 16, 8, semantics)
        } else if header_offset == 16 {
            // buffer
            copy_context.alloc_copy(from, bytes, 16, 16, semantics)
        } else {
            panic!(
                "unimplemented header_offset = {}, from = {:?}, size = {}",
                header_offset, from, bytes
            );
        };
        // `alloc_copy` should never return zero.
        debug_assert!(!dst.is_zero());

        let src = from_start;
        unsafe {
            std::ptr::copy_nonoverlapping::<u8>(src.to_ptr(), dst.to_mut_ptr(), bytes);
        }
        let to_obj = unsafe { ObjectReference::from_raw_address_unchecked(dst + header_offset) };

        copy_context.post_copy(to_obj, bytes, semantics);

        trace!("Copied object {} into {}", from, to_obj);

        unsafe {
            // GC-only path: use forwarding-aware type resolution because
            // the from object's vtag may point to a forwarded DataType.
            let vtag = mmtk_jl_typetagof(from.to_raw_address());
            if vtag.as_usize()
                >= ((crate::julia_types::jl_small_typeof_tags_jl_max_tags as usize) << 4)
            {
                if crate::julia_scanning::is_valid_datatype(vtag) {
                    let vt = crate::julia_scanning::mmtk_jl_to_typeof_resolving(vtag);
                    if !vt.is_null() && (*vt).name == jl_genericmemory_typename {
                        jl_gc_update_inlined_array(from.to_raw_address(), to_obj.to_raw_address());
                    }
                }
            }
        }

        // zero from_obj (for debugging purposes)
        #[cfg(debug_assertions)]
        {
            use atomic::Ordering;
            unsafe {
                libc::memset(from_start.to_mut_ptr(), 0, bytes);
            }

            Self::LOCAL_FORWARDING_BITS_SPEC.store_atomic::<JuliaVM, u8>(
                from,
                0b10_u8, // BEING_FORWARDED
                None,
                Ordering::SeqCst,
            );
        }

        to_obj
    }

    fn copy_to(_from: ObjectReference, _to: ObjectReference, _region: Address) -> Address {
        unimplemented!()
    }

    fn get_current_size(object: ObjectReference) -> usize {
        // Large pointer-bearing GenericMemory buffers are allocated inline
        // (how==0) in LOS so their slots carry RC/unlog side metadata (the LXR
        // invariant fix — see plan-alloc.md).  LXR's nursery-scan/promotion
        // path calls get_size() on such LOS objects, so this is legitimately
        // reached for LOS objects now; get_so_object_size handles them.
        unsafe { get_so_object_size(object) }
    }

    fn get_size_when_copied(_object: ObjectReference) -> usize {
        unimplemented!()
    }

    fn get_align_when_copied(_object: ObjectReference) -> usize {
        unimplemented!()
    }

    fn get_align_offset_when_copied(_object: ObjectReference) -> usize {
        unimplemented!()
    }

    fn get_reference_when_copied_to(_from: ObjectReference, _to: Address) -> ObjectReference {
        unimplemented!()
    }

    fn get_type_descriptor(_reference: ObjectReference) -> &'static [i8] {
        unimplemented!()
    }

    #[inline(always)]
    fn ref_to_object_start(object: ObjectReference) -> Address {
        if is_object_in_los(&object) {
            object.to_raw_address() - 48
        } else {
            unsafe { get_object_start_ref(object) }
        }
    }

    #[inline(always)]
    fn ref_to_header(object: ObjectReference) -> Address {
        object.to_raw_address()
    }

    fn dump_object(_object: ObjectReference) {
        unimplemented!()
    }

    fn dump_object_s(_object: ObjectReference) -> String {
        String::from("<julia object>")
    }

    fn get_class_pointer(object: ObjectReference) -> Address {
        // Return the Julia type tag pointer (jl_datatype_t*) for this object.
        // LXR's concurrent marking caches this to avoid re-reading the header
        // when scanning chunked large arrays during RC cascade.
        // GC-only path (called by LXR concurrent marking for klass caching):
        // use forwarding-aware resolution.
        unsafe { Address::from_usize(mmtk_jl_typeof_resolving(object.to_raw_address()) as usize) }
    }
}

#[inline(always)]
pub fn is_object_in_los(object: &ObjectReference) -> bool {
    // FIXME: get the range from MMTk. Or at least assert at boot time to make sure those constants are correct.
    (*object).to_raw_address().as_usize() >= 0x600_0000_0000
        && (*object).to_raw_address().as_usize() < 0x800_0000_0000
}

#[inline(always)]
/// This function uses mutable static variables and requires unsafe annotation
pub unsafe fn get_so_object_size(object: ObjectReference) -> usize {
    let obj_address = object.to_raw_address();
    let mut vtag = mmtk_jl_typetagof(obj_address);
    let mut vtag_usize = vtag.as_usize();

    if vtag_usize == JULIA_BUFF_TAG {
        return mmtk_get_obj_size(object);
    }

    if vtag_usize == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_unionall_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_uniontype_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_tvar_tag as usize) << 4)
        || vtag_usize == ((jl_small_typeof_tags_jl_vararg_tag as usize) << 4)
    {
        // these objects have pointers in them, but no other special handling
        // and we know their exact constant sizes:
        let dtsz = if vtag_usize == ((jl_small_typeof_tags_jl_datatype_tag as usize) << 4) {
            std::mem::size_of::<jl_datatype_t>()
        } else if vtag_usize == ((jl_small_typeof_tags_jl_unionall_tag as usize) << 4) {
            24
        } else if vtag_usize == ((jl_small_typeof_tags_jl_uniontype_tag as usize) << 4) {
            16
        } else if vtag_usize == ((jl_small_typeof_tags_jl_tvar_tag as usize) << 4) {
            24
        } else {
            32 // Vararg
        };

        return llt_align(dtsz + JULIA_HEADER_SIZE, 16);
    } else if vtag_usize < ((jl_small_typeof_tags_jl_max_tags as usize) << 4) {
        if vtag_usize == ((jl_small_typeof_tags_jl_simplevector_tag as usize) << 4) {
            let length = (*obj_address.to_ptr::<jl_svec_t>()).length;
            let dtsz = length * std::mem::size_of::<Address>() + std::mem::size_of::<jl_svec_t>();

            debug_assert!(
                dtsz + JULIA_HEADER_SIZE <= 2032,
                "size {} greater than minimum!",
                dtsz + JULIA_HEADER_SIZE
            );

            return llt_align(dtsz + JULIA_HEADER_SIZE, 16);
        } else if vtag_usize == ((jl_small_typeof_tags_jl_module_tag as usize) << 4) {
            let dtsz = std::mem::size_of::<jl_module_t>();
            debug_assert!(
                dtsz + JULIA_HEADER_SIZE <= 2032,
                "size {} greater than minimum!",
                dtsz + JULIA_HEADER_SIZE
            );

            return llt_align(dtsz + JULIA_HEADER_SIZE, 16);
        } else if vtag_usize == ((jl_small_typeof_tags_jl_task_tag as usize) << 4) {
            let dtsz = std::mem::size_of::<jl_task_t>();
            debug_assert!(
                dtsz + JULIA_HEADER_SIZE <= 2032,
                "size {} greater than minimum!",
                dtsz + JULIA_HEADER_SIZE
            );

            return llt_align(dtsz + JULIA_HEADER_SIZE, 16);
        } else if vtag_usize == ((jl_small_typeof_tags_jl_string_tag as usize) << 4) {
            let length = object.to_raw_address().load::<usize>();
            let dtsz = length + std::mem::size_of::<usize>() + 1;

            debug_assert!(
                dtsz + JULIA_HEADER_SIZE <= 2032,
                "size {} greater than minimum!",
                dtsz + JULIA_HEADER_SIZE
            );

            // NB: Strings are aligned to 8 and not to 16
            return llt_align(dtsz + JULIA_HEADER_SIZE, 8);
        } else {
            let vt = jl_small_typeof[vtag_usize / std::mem::size_of::<Address>()];
            if vt.is_null() {
                return llt_align(JULIA_HEADER_SIZE, 16);
            }
            let layout = (*vt).layout;
            if layout.is_null() || (layout as usize) < 0x20000 {
                return llt_align(JULIA_HEADER_SIZE, 16);
            }
            let dtsz = (*layout).size as usize;
            debug_assert!(
                dtsz + JULIA_HEADER_SIZE <= 2032,
                "size {} greater than minimum!",
                dtsz + JULIA_HEADER_SIZE
            );

            return llt_align(dtsz + JULIA_HEADER_SIZE, 16);
        }
    } else {
        // vtag is a direct pointer to a jl_datatype_t in the heap.
        // During nursery evacuation, this DataType may itself have been
        // forwarded — its header is now a forwarding pointer, not a type tag.
        // Resolve through the forwarding chain before validation.
        vtag = resolve_forwarded_datatype_addr(vtag);

        let is_datatype = crate::julia_scanning::is_valid_datatype(vtag);

        if !is_datatype {
            let mut status = 0u8;
            if crate::collection::is_gc_thread() {
                // DataType is a regular object at offset +8, so we must restore bit 3
                let obj_ref_addr = Address::from_usize(vtag.as_usize() | 8);
                if crate::api::mmtk_object_is_managed_by_mmtk(vtag.as_usize()) {
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
            return llt_align(JULIA_HEADER_SIZE, 16);
        }

        let vt = if vtag.as_usize() < 0x20000 {
            unsafe { crate::julia_scanning::safe_jl_datatype_type() }
        } else {
            vtag.to_ptr::<jl_datatype_t>()
        };
        let type_tag = mmtk_jl_typetagof(vtag);
        let type_tag_usize = type_tag.as_usize();
        let datatype_type_addr =
            unsafe { crate::julia_scanning::jl_mmtk_get_jl_datatype_type_addr() }.as_usize();
        let datatype_type_val = if datatype_type_addr != 0 {
            unsafe { Address::from_usize(datatype_type_addr).load::<usize>() }
        } else {
            0
        };
    }

    let vt = if vtag.as_usize() < 0x20000 {
        unsafe { crate::julia_scanning::safe_jl_datatype_type() }
    } else {
        vtag.to_ptr::<jl_datatype_t>()
    };
    if vt.is_null() {
        return llt_align(512, 16);
    }
    if (*vt).name == jl_genericmemory_typename {
        let m = obj_address.to_ptr::<jl_genericmemory_t>();
        let how = jl_gc_genericmemory_how(obj_address);
        let res = if how == 0 {
            // Use already-resolved `vt` instead of re-reading the raw header,
            // which would fail for forwarded DataTypes.
            let layout = (*vt).layout;
            let mut sz = (*layout).size as usize * (*m).length;
            if (*layout).flags.arrayelem_isunion() != 0 {
                sz += (*m).length;
            }

            let dtsz = llt_align(std::mem::size_of::<jl_genericmemory_t>(), 16);
            llt_align(sz + dtsz + JULIA_HEADER_SIZE, 16)
        } else {
            let dtsz = std::mem::size_of::<jl_genericmemory_t>() + std::mem::size_of::<Address>();
            llt_align(dtsz + JULIA_HEADER_SIZE, 16)
        };

        // Inline (how==0) pointer-bearing buffers may exceed the Immix size
        // class and live in LOS (see plan-alloc.md); the <=2032 bound only
        // applies to small-object-space residents.
        debug_assert!(
            res <= 2032 || is_object_in_los(&object),
            "size {} greater than minimum!",
            res
        );

        return res;
    }

    let layout = (*vt).layout;
    if layout.is_null() || (layout as usize) < 0x20000 {
        return llt_align(JULIA_HEADER_SIZE, 16);
    }
    let dtsz = (*layout).size as usize;
    debug_assert!(
        dtsz + JULIA_HEADER_SIZE <= 2032,
        "size {} greater than minimum!",
        dtsz + JULIA_HEADER_SIZE
    );

    llt_align(dtsz + JULIA_HEADER_SIZE, 16)
}

#[inline(always)]
pub unsafe fn get_object_start_ref(object: ObjectReference) -> Address {
    let obj_address = object.to_raw_address();
    let obj_type = mmtk_jl_typeof(obj_address);

    if obj_type as usize == JULIA_BUFF_TAG {
        obj_address - 2 * JULIA_HEADER_SIZE
    } else {
        obj_address - JULIA_HEADER_SIZE
    }
}

#[inline(always)]
pub unsafe fn llt_align(size: usize, align: usize) -> usize {
    ((size) + (align) - 1) & !((align) - 1)
}

#[inline(always)]
pub unsafe fn mmtk_jl_is_uniontype(t: *const jl_datatype_t) -> bool {
    mmtk_jl_typetagof(Address::from_ptr(t)).as_usize()
        == (jl_small_typeof_tags_jl_uniontype_tag << 4) as usize
}
