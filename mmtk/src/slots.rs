use atomic::Atomic;
use mmtk::{
    util::{Address, ObjectReference},
    vm::{
        slot::{SimpleSlot, Slot},
        RootsWorkFactory,
    },
};

/// If a VM supports multiple kinds of slots, we can use tagged union to represent all of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum JuliaVMSlot {
    Simple(SimpleSlot),
    Offset(OffsetSlot),
    /// Carries a captured object reference directly, without an underlying
    /// memory slot.  Used for non-heap slot barriers (external GenericMemory
    /// data with how!=0) where the new value is known at barrier time.
    ///
    /// At GC time, `load()` returns the captured value without any memory
    /// read — this avoids re-reading a non-heap slot that may have been
    /// overwritten (which would cause RC overcount from duplicate incs).
    ///
    /// `to_address()` returns `Address::ZERO` so that:
    /// - `unlog_and_load_rc_object` skips `unlog_field_relaxed` (heap range guard)
    /// - `record_mature_evac_remset` skips (address_in_defrag → false)
    /// - `store()` is a no-op (no slot to update)
    Direct(ObjectReference),
    TypeTag(TypeTagSlot),
}

unsafe impl Send for JuliaVMSlot {}

impl Slot for JuliaVMSlot {
    fn load(&self) -> Option<ObjectReference> {
        match self {
            JuliaVMSlot::Simple(e) => e.load(),
            JuliaVMSlot::Offset(e) => e.load(),
            JuliaVMSlot::Direct(o) => Some(*o),
            JuliaVMSlot::TypeTag(e) => e.load(),
        }
    }

    fn store(&self, object: ObjectReference) {
        match self {
            JuliaVMSlot::Simple(e) => {
                unsafe {
                    crate::julia_scanning::update_array_ptr_or_offset_if_needed(
                        e.as_address(),
                        object,
                    );
                }
                e.store(object);
            }
            JuliaVMSlot::Offset(e) => e.store(object),
            // No-op: the actual non-heap slot is written by C code after
            // the barrier returns.  Direct slots exist only to carry the
            // captured value through the incs queue without re-reading.
            JuliaVMSlot::Direct(_) => {}
            JuliaVMSlot::TypeTag(e) => e.store(object),
        }
    }

    fn to_address(&self) -> Address {
        match self {
            JuliaVMSlot::Simple(e) => e.as_address(),
            JuliaVMSlot::Offset(e) => e.slot_address(),
            // Sentinel: Address::ZERO is outside the heap range, so all
            // heap-range-guarded operations (unlog_field_relaxed,
            // address_in_defrag, record_mature_evac_remset) skip cleanly.
            JuliaVMSlot::Direct(_) => Address::ZERO,
            JuliaVMSlot::TypeTag(e) => e.address,
        }
    }

    fn raw_address(&self) -> Address {
        match self {
            JuliaVMSlot::Simple(e) => e.as_address(),
            JuliaVMSlot::Offset(e) => e.slot_address(),
            JuliaVMSlot::Direct(_) => Address::ZERO,
            JuliaVMSlot::TypeTag(e) => e.address,
        }
    }

    fn from_address(addr: Address) -> Self {
        JuliaVMSlot::Simple(SimpleSlot::from_address(addr))
    }

    #[inline(always)]
    fn is_type_tag(&self) -> bool {
        matches!(self, JuliaVMSlot::TypeTag(_))
    }
}

impl std::fmt::Debug for JuliaVMSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple(e) => write!(f, "{}", e.as_address()),
            Self::Offset(e) => write!(f, "{}+{}", e.slot_address(), e.offset),
            Self::Direct(o) => write!(f, "Direct({:?})", o),
            Self::TypeTag(e) => write!(f, "TypeTag({})", e.address),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeTagSlot {
    pub address: Address,
}

impl TypeTagSlot {
    pub fn load(&self) -> Option<ObjectReference> {
        let value = unsafe { self.address.load::<Address>() };
        let value_usize = value.as_usize();
        if value_usize == 0 {
            None
        } else {
            // Mask out the lower 4 bits (GC/status bits)
            let type_addr = unsafe { Address::from_usize(value_usize & !0xf) };
            unsafe { ObjectReference::from_raw_address(type_addr) }
        }
    }

    pub fn store(&self, object: ObjectReference) {
        let old_value = unsafe { self.address.load::<usize>() };
        // Preserve the lower 4 bits (GC/status bits)
        let gc_bits = old_value & 0xf;
        let new_value = object.to_raw_address().as_usize() | gc_bits;
        unsafe { self.address.store::<usize>(new_value) };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OffsetSlot {
    slot_addr: *mut Atomic<Address>,
    offset: usize,
}

unsafe impl Send for OffsetSlot {}

impl OffsetSlot {
    pub fn new_no_offset(address: Address) -> Self {
        Self {
            slot_addr: address.to_mut_ptr(),
            offset: 0,
        }
    }

    pub fn new_with_offset(address: Address, offset: usize) -> Self {
        Self {
            slot_addr: address.to_mut_ptr(),
            offset,
        }
    }

    pub fn slot_address(&self) -> Address {
        Address::from_mut_ptr(self.slot_addr)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl Slot for OffsetSlot {
    fn load(&self) -> Option<ObjectReference> {
        let middle = unsafe { (*self.slot_addr).load(atomic::Ordering::Relaxed) };
        let begin = middle - self.offset;
        debug_assert!(!begin.is_zero());
        ObjectReference::from_raw_address(begin)
    }

    fn store(&self, object: ObjectReference) {
        let begin = object.to_raw_address();
        let middle = begin + self.offset;
        unsafe { (*self.slot_addr).store(middle, atomic::Ordering::Relaxed) }
    }
}

#[derive(Hash, Clone, PartialEq, Eq, Debug)]
pub struct JuliaMemorySlice {
    pub owner: ObjectReference,
    pub start: Address,
    pub count: usize,
}

impl mmtk::vm::slot::MemorySlice for JuliaMemorySlice {
    type SlotType = JuliaVMSlot;
    type SlotIterator = JuliaMemorySliceSlotIterator;
    type ChunkIterator = std::vec::IntoIter<Self>;

    fn iter_slots(&self) -> Self::SlotIterator {
        JuliaMemorySliceSlotIterator {
            cursor: self.start,
            limit: self.start.shift::<Address>(self.count as isize),
        }
    }

    fn object(&self) -> Option<ObjectReference> {
        Some(self.owner)
    }

    fn start(&self) -> Address {
        self.start
    }

    fn bytes(&self) -> usize {
        self.count << mmtk::util::constants::LOG_BYTES_IN_ADDRESS
    }

    fn chunks(&self, chunk_size: usize) -> Self::ChunkIterator {
        let total_slots = self.count;
        let mut chunks = vec![];
        let mut offset = 0;
        while offset < total_slots {
            let chunk_count = std::cmp::min(chunk_size, total_slots - offset);
            chunks.push(JuliaMemorySlice {
                owner: self.owner,
                start: self.start.shift::<Address>(offset as isize),
                count: chunk_count,
            });
            offset += chunk_count;
        }
        chunks.into_iter()
    }

    fn len(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Self::SlotType {
        use mmtk::vm::slot::SimpleSlot;
        JuliaVMSlot::Simple(SimpleSlot::from_address(
            self.start.shift::<Address>(index as isize),
        ))
    }

    fn copy(src: &Self, tgt: &Self) {
        use std::sync::atomic::*;
        // Raw memory copy -- we should be consistent with jl_array_ptr_copy in array.c
        unsafe {
            let words = tgt.bytes() >> mmtk::util::constants::LOG_BYTES_IN_ADDRESS;
            // let src = src.start().to_ptr::<usize>();
            // let tgt = tgt.start().to_mut_ptr::<usize>();
            // std::ptr::copy(src, tgt, words)

            let src_addr = src.start();
            let tgt_addr = tgt.start();

            let n: isize = words as isize;

            if tgt_addr < src_addr || tgt_addr > src_addr + tgt.bytes() {
                // non overlaping
                for i in 0..n {
                    let val: usize = src_addr
                        .shift::<usize>(i)
                        .atomic_load::<AtomicUsize>(Ordering::Relaxed);
                    tgt_addr
                        .shift::<usize>(i)
                        .atomic_store::<AtomicUsize>(val, Ordering::Release);
                }
            } else {
                for i in 0..n {
                    let val = src_addr
                        .shift::<usize>(n - i - 1)
                        .atomic_load::<AtomicUsize>(Ordering::Relaxed);
                    tgt_addr
                        .shift::<usize>(n - i - 1)
                        .atomic_store::<AtomicUsize>(val, Ordering::Release);
                }
            }
        }
    }
}

pub struct JuliaMemorySliceSlotIterator {
    cursor: Address,
    limit: Address,
}

impl Iterator for JuliaMemorySliceSlotIterator {
    type Item = JuliaVMSlot;

    fn next(&mut self) -> Option<JuliaVMSlot> {
        if self.cursor >= self.limit {
            None
        } else {
            let slot = self.cursor;
            self.cursor = self.cursor.shift::<ObjectReference>(1);
            Some(JuliaVMSlot::Simple(SimpleSlot::from_address(slot)))
        }
    }
}

const ROOT_WORK_PACKET_SIZE: usize = 4096;

#[repr(C)]
pub struct RootsWorkBuffer<T: Copy> {
    pub ptr: *mut T,
    pub capacity: usize,
}

impl<T: Copy> RootsWorkBuffer<T> {
    fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            capacity: 0,
        }
    }
    fn new() -> Self {
        let (buf, _, capacity) = {
            let new_vec = Vec::with_capacity(ROOT_WORK_PACKET_SIZE);
            let mut me = std::mem::ManuallyDrop::new(new_vec);
            (me.as_mut_ptr(), me.len(), me.capacity())
        };
        Self { ptr: buf, capacity }
    }
}

#[repr(C)]
pub struct RootsWorkClosure {
    pub report_slots_func: extern "C" fn(
        buf: *mut Address,
        size: usize,
        cap: usize,
        factory_ptr: *mut libc::c_void,
        renew: bool,
    ) -> RootsWorkBuffer<Address>,
    pub report_nodes_func: extern "C" fn(
        buf: *mut ObjectReference,
        size: usize,
        cap: usize,
        factory_ptr: *mut libc::c_void,
        renew: bool,
    ) -> RootsWorkBuffer<ObjectReference>,
    pub report_tpinned_nodes_func: extern "C" fn(
        buf: *mut ObjectReference,
        size: usize,
        cap: usize,
        factory_ptr: *mut libc::c_void,
        renew: bool,
    ) -> RootsWorkBuffer<ObjectReference>,
    pub factory_ptr: *mut libc::c_void,
}

impl RootsWorkClosure {
    extern "C" fn report_simple_slots<F: RootsWorkFactory<JuliaVMSlot>>(
        buf: *mut Address,
        size: usize,
        cap: usize,
        factory_ptr: *mut libc::c_void,
        renew: bool,
    ) -> RootsWorkBuffer<Address> {
        if !buf.is_null() {
            let buf = unsafe { Vec::<Address>::from_raw_parts(buf, size, cap) }
                .into_iter()
                .map(|addr| JuliaVMSlot::Simple(SimpleSlot::from_address(addr)))
                .collect();
            let factory: &mut F = unsafe { &mut *(factory_ptr as *mut F) };
            factory.create_process_roots_work(buf, mmtk::scheduler::RootKind::Strong);
        }

        if renew {
            RootsWorkBuffer::new()
        } else {
            RootsWorkBuffer::empty()
        }
    }

    extern "C" fn report_nodes<F: RootsWorkFactory<JuliaVMSlot>>(
        buf: *mut ObjectReference,
        size: usize,
        cap: usize,
        factory_ptr: *mut libc::c_void,
        renew: bool,
    ) -> RootsWorkBuffer<ObjectReference> {
        if !buf.is_null() {
            let buf = unsafe { Vec::<ObjectReference>::from_raw_parts(buf, size, cap) };
            let factory: &mut F = unsafe { &mut *(factory_ptr as *mut F) };
            factory.create_process_pinning_roots_work(buf);
        }

        if renew {
            RootsWorkBuffer::new()
        } else {
            RootsWorkBuffer::empty()
        }
    }

    extern "C" fn report_tpinned_nodes<F: RootsWorkFactory<JuliaVMSlot>>(
        buf: *mut ObjectReference,
        size: usize,
        cap: usize,
        factory_ptr: *mut libc::c_void,
        renew: bool,
    ) -> RootsWorkBuffer<ObjectReference> {
        if !buf.is_null() {
            let buf = unsafe { Vec::<ObjectReference>::from_raw_parts(buf, size, cap) };
            let factory: &mut F = unsafe { &mut *(factory_ptr as *mut F) };
            factory.create_process_tpinning_roots_work(buf);
        }

        if renew {
            RootsWorkBuffer::new()
        } else {
            RootsWorkBuffer::empty()
        }
    }

    pub fn from_roots_work_factory<F: RootsWorkFactory<JuliaVMSlot>>(factory: &mut F) -> Self {
        RootsWorkClosure {
            report_slots_func: Self::report_simple_slots::<F>,
            report_nodes_func: Self::report_nodes::<F>,
            report_tpinned_nodes_func: Self::report_tpinned_nodes::<F>,
            factory_ptr: factory as *mut F as *mut libc::c_void,
        }
    }
}
