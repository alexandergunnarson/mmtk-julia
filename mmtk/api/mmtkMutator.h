#ifndef MMTK_JULIA_MMTK_MUTATOR_H
#define MMTK_JULIA_MMTK_MUTATOR_H

// mmtk_julia_types.h refers to the types in this file.
// So if this file is updated, make sure you regenerate Rust types for mmtk_julia_types.h.
//
// NOTE: This header must match the #[repr(C)] Rust struct layouts in mmtk-core.
// If using the LXR fork (wenyuzhao/mmtk-core), ImmixAllocator has additional fields
// for block recycling, local block lists, etc.  The struct sizes were verified against
// Rust's reported sizeof(Mutator<JuliaVM>) = 992 bytes.

enum Allocator {
  AllocatorDefault = 0,
  AllocatorImmortal = 1,
  AllocatorLos = 2,
  AllocatorCode = 3,
  AllocatorReadOnly = 4,
};

typedef struct {
  void* data;
  void* vtable;
} RustDynPtr;

// These constants should match the constants defined in mmtk::util::alloc::allocators
#define MAX_BUMP_ALLOCATORS 6
#define MAX_LARGE_OBJECT_ALLOCATORS 2
#define MAX_MALLOC_ALLOCATORS 1
#define MAX_IMMIX_ALLOCATORS 2
#define MAX_FREE_LIST_ALLOCATORS 2
#define MAX_MARK_COMPACT_ALLOCATORS 1

// The following types should have the same layout as the types with the same name in MMTk core (Rust)

// mmtk::util::alloc::bumpallocator::BumpPointer
// #[repr(C)]: cursor: Address (usize), limit: Address (usize)
typedef struct {
  void* cursor;
  void* limit;
} BumpPointer;

// mmtk::util::alloc::bumpallocator::BumpAllocator<VM>
// #[repr(C)]: tls, bump_pointer, space (RustDynPtr), context
typedef struct {
  void* tls;
  void* cursor;   // bump_pointer.cursor
  void* limit;    // bump_pointer.limit
  RustDynPtr space;
  void* context;
} BumpAllocator;

// mmtk::util::alloc::large_object_allocator::LargeObjectAllocator<VM>
typedef struct {
  void* tls;
  void* space;
  void* context;
} LargeObjectAllocator;

// mmtk::util::alloc::immix_allocator::ImmixAllocator<VM>  (LXR fork)
// #[repr(C)] — must match field order in wenyuzhao/mmtk-core ImmixAllocator
// Total size: 192 bytes (verified: 2 * (192 - 88) = 208 = Mutator size difference)
typedef struct {
  void*    tls;                              // VMThread (OpaquePointer = usize)        off=0
  void*    cursor;                           // bump_pointer.cursor                     off=8
  void*    limit;                            // bump_pointer.limit                      off=16
  void*    immix_space;                      // &'static ImmixSpace<VM>                 off=24
  void*    context;                          // Arc<AllocatorContext<VM>>                off=32
  uint8_t  hot;                              // bool                                    off=40
  uint8_t  copy;                             // bool                                    off=41
  uint8_t  _pad1[6];                         // padding to 8-byte align                 off=42
  void*    large_cursor;                     // large_bump_pointer.cursor                off=48
  void*    large_limit;                      // large_bump_pointer.limit                 off=56
  uint8_t  request_for_large;                // bool                                    off=64
  uint8_t  _pad2[7];                         // padding to 8-byte align                 off=65
  // Option<Line> = discriminant (8 bytes padded) + Address (8 bytes)
  uint8_t  line_tag;                         // Option discriminant (0=None, 1=Some)    off=72
  uint8_t  _pad_line[7];                     // padding                                 off=73
  uintptr_t line_val;                        // Line (Address = usize)                  off=80
  // Option<Block>
  uint8_t  block_tag;                        // Option discriminant                     off=88
  uint8_t  _pad_block[7];                    // padding                                 off=89
  uintptr_t block_val;                       // Block (Address = usize)                 off=96
  // Option<Block>
  uint8_t  large_block_tag;                  // Option discriminant                     off=104
  uint8_t  _pad_large_block[7];              // padding                                 off=105
  uintptr_t large_block_val;                 // Block (Address = usize)                 off=112
  void*    mutator_recycled_blocks;          // Box<Vec<Block>>                         off=120
  void*    local_clean_blocks;               // Box<Vec<Block>>                         off=128
  void*    local_reuse_blocks;               // Box<Vec<Block>>                         off=136
  uintptr_t local_clean_blocks_cursor;       // usize                                   off=144
  uintptr_t local_clean_blocks_cursor_boundary; // usize                                off=152
  uintptr_t local_reuse_blocks_cursor;       // usize                                   off=160
  uintptr_t local_reuse_blocks_cursor_boundary; // usize                                off=168
  uintptr_t mutator_recycled_lines;          // usize                                   off=176
  uint8_t  retry;                            // bool                                    off=184
  uint8_t  _pad3[7];                         // padding to 8-byte struct alignment      off=185
} ImmixAllocator;                            // total = 192 bytes

typedef struct {
  void* Address;
} FLBlock;

typedef struct {
  FLBlock first;
  FLBlock last;
  size_t size;
  char lock;
} FLBlockList;

typedef struct {
  void* tls;
  void* space;
  void* context;
  FLBlockList* available_blocks;
  FLBlockList* available_blocks_stress;
  FLBlockList* unswept_blocks;
  FLBlockList* consumed_blocks;
} FreeListAllocator;

typedef struct {
  void* tls;
  void* space;
  void* context;
} MMTkMallocAllocator; // Prefix with MMTk to avoid name clash

typedef struct {
  BumpAllocator bump_allocator;
} MarkCompactAllocator;

typedef struct {
  BumpAllocator bump_pointer[MAX_BUMP_ALLOCATORS];
  LargeObjectAllocator large_object[MAX_LARGE_OBJECT_ALLOCATORS];
  MMTkMallocAllocator malloc[MAX_MALLOC_ALLOCATORS];
  ImmixAllocator immix[MAX_IMMIX_ALLOCATORS];
  FreeListAllocator free_list[MAX_FREE_LIST_ALLOCATORS];
  MarkCompactAllocator markcompact[MAX_MARK_COMPACT_ALLOCATORS];
} Allocators;

typedef struct {
  void* allocator_mapping;
  void* space_mapping;
  RustDynPtr prepare_func;
  RustDynPtr release_func;
} MutatorConfig;

typedef struct {
  Allocators allocators;
  RustDynPtr barrier;
  void* mutator_tls;
  RustDynPtr plan;
  MutatorConfig config;
} MMTkMutatorContext;

// Compile-time size check (only works in C11+)
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(ImmixAllocator) == 192, "ImmixAllocator size mismatch with Rust");
_Static_assert(sizeof(MMTkMutatorContext) == 992, "MMTkMutatorContext size mismatch with Rust");
#endif

#endif // MMTK_JULIA_MMTK_MUTATOR_H
