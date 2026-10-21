// memshare.cpp - Lab 3, Task 12 Part B.
//
// Give the NPU a buffer it can MAP instead of one it has to COPY.
//
// The harness (matmul_qnn_shared_spike) makes ONE allocation big enough for all
// three tensors and registers three regions inside it at different offsets. It
// asks you for that allocation through three C symbols:
//
//   void* AllocateSharedBuffer(size_t size);   // NULL on failure
//   void  FreeSharedBuffer(void* ptr);
//   int   GetSharedBufferFd(void* ptr);        // >= 0 real FD; -1 = not shareable
//
// The file descriptor is the whole point. QNN's HTP shared-buffer path takes an
// FD plus an offset, not a pointer - so a buffer with no FD cannot be shared,
// and the harness will refuse the run rather than silently fall back.
//
// As shipped this file allocates ordinary heap memory and returns -1 for the FD.
// That builds and loads, so nothing is blocked while you work, but the harness
// will stop with "the buffer is not exportable". Set MEMSHARE_STUB_ONLY to 0 and
// fill in the five TODOs to get the real path.
//
// On this board the rpcmem symbols live in /vendor/lib64/libcdsprpc.so.
// There is no librpcmem.so here; trying it first is harmless and portable.
//
// Build (from the repo root); this also pushes it to the board:
//   bash scripts/build_student.sh memshare
//
// Then compare a baseline run against a shared one:
//   python profiler.py matmul npu --variant spike -p size=1024
//   python profiler.py matmul npu --variant spike -p size=1024 -p memshare=./libmemshare.so
//
// The shared run must show `shared_impl=memshare`. If it shows builtin_rpcmem
// the harness allocated the memory itself and your code never ran.

#include <dlfcn.h>     // dlopen, dlsym
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>    // getpagesize
#include <mutex>
#include <atomic>

// -----------------------------------------------------------------------------
// SWITCH: the lab should not block before you finish the TODOs.
// Keep MEMSHARE_STUB_ONLY = 1 while you implement. When you are done,
// set it to 0 to enable the rpcmem path.
// -----------------------------------------------------------------------------
#define MEMSHARE_STUB_ONLY 1

// Round up to the OS page size (helps both fallback and rpcmem).
static size_t page_align(size_t n) {
  size_t p = (size_t)getpagesize();
  return (n + p - 1) & ~(p - 1);
}

extern "C" __attribute__((visibility("default")))
void* AllocateSharedBuffer(size_t size);

extern "C" __attribute__((visibility("default")))
void FreeSharedBuffer(void* ptr);

extern "C" __attribute__((visibility("default")))
int GetSharedBufferFd(void* ptr);

#if MEMSHARE_STUB_ONLY
// ============================================================================
// FALLBACK MODE (works today, no zero-copy FD)
// ----------------------------------------------------------------------------
// You are not using rpcmem here. This guarantees the assignment runs end-to-end
// even on devices without rpcmem or before you finish the TODOs.
// ============================================================================

extern "C" __attribute__((visibility("default")))
void* AllocateSharedBuffer(size_t size) {
  size = page_align(size);
  void* ptr = nullptr;
  if (posix_memalign(&ptr, (size_t)getpagesize(), size) != 0) return nullptr;

  static std::once_flag once;
  std::call_once(once, [](){
    fprintf(stderr, "[memshare] Fallback allocator active (no rpcmem / no FD)\n");
  });
  return ptr;
}

extern "C" __attribute__((visibility("default")))
void FreeSharedBuffer(void* ptr) {
  if (!ptr) return;
  free(ptr);
}

extern "C" __attribute__((visibility("default")))
int GetSharedBufferFd(void* /*ptr*/) {
  // No file descriptor in fallback mode
  return -1;
}

#else
// ============================================================================
// ZERO-COPY MODE (you fill the TODOs)
// ----------------------------------------------------------------------------
// Context: Qualcomm's rpcmem allocator can return a pointer that also maps to
// a shareable file descriptor (FD). QNN's HTP "Shared Buffer" path expects
// that FD so the DSP can directly access the memory without extra copies.
// ============================================================================

// Function pointer types for the rpcmem API you will resolve at runtime.
// Tip: search for "rpcmem.h" on your device/vendor sources or docs to confirm.
using rpcmem_init_t    = void (*)();
using rpcmem_deinit_t  = void (*)();
using rpcmem_alloc_t   = void* (*)(int heapid, uint32_t flags, int size);
using rpcmem_free_t    = void  (*)(void*);
using rpcmem_to_fd_t   = int   (*)(void*);

// Globals for the dynamically loaded library and symbols.
static void*             g_rpcmem_so     = nullptr;
static rpcmem_init_t     p_rpcmem_init   = nullptr;
static rpcmem_deinit_t   p_rpcmem_deinit = nullptr;
static rpcmem_alloc_t    p_rpcmem_alloc  = nullptr;
static rpcmem_free_t     p_rpcmem_free   = nullptr;
static rpcmem_to_fd_t    p_rpcmem_to_fd  = nullptr;
static std::once_flag    g_once;
static std::atomic<bool> g_has_rpcmem{false};

// One-time loader:
//
// You should:
//  - try dlopen("librpcmem.so") first; if that fails, try dlopen("libcdsprpc.so").
//  - use dlsym to resolve: "rpcmem_init", "rpcmem_deinit", "rpcmem_alloc",
//                          "rpcmem_free", "rpcmem_to_fd".
//  - if alloc/free are present, call rpcmem_init() once (if it exists), set g_has_rpcmem=true.
//  - if a load attempt fails or required symbols are missing, dlclose any opened handle
//    and leave g_has_rpcmem=false so the code will fall back gracefully.
static void try_init_rpcmem() {
  // TODO(A): implement dynamic loading with dlopen/dlsym as described above.
  // Functions to look up with dlsym: rpcmem_init, rpcmem_deinit, rpcmem_alloc,
  //                                  rpcmem_free, rpcmem_to_fd.
}

extern "C" __attribute__((visibility("default")))
void* AllocateSharedBuffer(size_t size) {
  std::call_once(g_once, try_init_rpcmem);
  size = page_align(size);

  if (g_has_rpcmem.load()) {
    // TODO(B): allocate via rpcmem to get a real shared buffer and FD.
    // You should call p_rpcmem_alloc with:
    //   - heap  = -1  (let rpcmem choose a default heap)
    //   - flags = 1   (cached mapping works well for this lab)
    // If allocation succeeds, return the pointer you got from p_rpcmem_alloc.
  }

  // Fallback remains available even in zero-copy build, in case runtime load fails.
  void* ptr = nullptr;
  if (posix_memalign(&ptr, (size_t)getpagesize(), size) != 0) return nullptr;

  static std::once_flag once;
  std::call_once(once, [](){
    fprintf(stderr, "[memshare] rpcmem not available at runtime; using fallback\n");
  });
  return ptr;
}

extern "C" __attribute__((visibility("default")))
void FreeSharedBuffer(void* ptr) {
  if (!ptr) return;
  if (g_has_rpcmem.load() && p_rpcmem_free) {
    // TODO(C): free via rpcmem when the pointer came from rpcmem_alloc.
    // If you only call FreeSharedBuffer on blocks you allocated here, it is fine
    // to always call p_rpcmem_free when it is available.
    return;
  }
  free(ptr);
}

extern "C" __attribute__((visibility("default")))
int GetSharedBufferFd(void* ptr) {
  if (g_has_rpcmem.load() && p_rpcmem_to_fd) {
    // TODO(D): return the shareable file descriptor for this allocation.
    // You should call p_rpcmem_to_fd(ptr) and return its value.
    // A non-negative FD indicates true zero-copy support.
  }
  // Fallback: no FD available
  return -1;
}

// Optional cleanup:
// If you called rpcmem_init() during try_init_rpcmem(), you should
// call rpcmem_deinit() here to be polite. Not strictly required for the lab.
__attribute__((destructor))
static void memshare_destructor() {
  if (g_has_rpcmem.load() && p_rpcmem_deinit) {
    // TODO(E): call rpcmem_deinit().
  }
  if (g_rpcmem_so) dlclose(g_rpcmem_so);
}
#endif  // MEMSHARE_STUB_ONLY
