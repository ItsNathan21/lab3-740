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
// Set MEMSHARE_STUB_ONLY to 1 to exercise the heap-only baseline, which
// returns -1 when the harness asks for a file descriptor.
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
#include <limits.h>    // INT_MAX
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>    // getpagesize
#include <mutex>
#include <atomic>

// -----------------------------------------------------------------------------
// Enable the rpcmem path for the shared-buffer run.
// -----------------------------------------------------------------------------
#define MEMSHARE_STUB_ONLY 0

// Round up to the OS page size (helps both fallback and rpcmem).
static size_t page_align(size_t n) {
  const size_t p = (size_t)getpagesize();
  if (n == 0 || n > SIZE_MAX - (p - 1)) return 0;
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
  if (size == 0) return nullptr;
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
// ZERO-COPY MODE
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

static void try_init_rpcmem() {
  const char* libraries[] = {"librpcmem.so", "libcdsprpc.so"};
  for (const char* library : libraries) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!handle) continue;

    auto alloc = reinterpret_cast<rpcmem_alloc_t>(dlsym(handle, "rpcmem_alloc"));
    auto release = reinterpret_cast<rpcmem_free_t>(dlsym(handle, "rpcmem_free"));
    auto to_fd = reinterpret_cast<rpcmem_to_fd_t>(dlsym(handle, "rpcmem_to_fd"));
    if (!alloc || !release || !to_fd) {
      dlclose(handle);
      continue;
    }

    g_rpcmem_so = handle;
    p_rpcmem_alloc = alloc;
    p_rpcmem_free = release;
    p_rpcmem_to_fd = to_fd;
    p_rpcmem_init = reinterpret_cast<rpcmem_init_t>(dlsym(handle, "rpcmem_init"));
    p_rpcmem_deinit = reinterpret_cast<rpcmem_deinit_t>(dlsym(handle, "rpcmem_deinit"));
    if (p_rpcmem_init) p_rpcmem_init();
    g_has_rpcmem.store(true);
    return;
  }
}

extern "C" __attribute__((visibility("default")))
void* AllocateSharedBuffer(size_t size) {
  size = page_align(size);
  if (size == 0 || size > INT_MAX) return nullptr;
  std::call_once(g_once, try_init_rpcmem);

  if (g_has_rpcmem.load()) {
    // Keep a single allocation backend for the lifetime of the library.
    // This lets FreeSharedBuffer identify the allocator for every live pointer.
    constexpr int default_heap = -1;
    constexpr int system_heap = 25;
    constexpr uint32_t cached = 1;
    void* ptr = p_rpcmem_alloc(default_heap, cached, static_cast<int>(size));
    return ptr ? ptr : p_rpcmem_alloc(system_heap, cached, static_cast<int>(size));
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
  if (g_has_rpcmem.load()) p_rpcmem_free(ptr);
  else free(ptr);
}

extern "C" __attribute__((visibility("default")))
int GetSharedBufferFd(void* ptr) {
  if (!ptr || !g_has_rpcmem.load()) return -1;
  const int fd = p_rpcmem_to_fd(ptr);
  return fd < 0 ? -1 : fd;
}

__attribute__((destructor))
static void memshare_destructor() {
  if (p_rpcmem_init && p_rpcmem_deinit) p_rpcmem_deinit();
  if (g_rpcmem_so) dlclose(g_rpcmem_so);
}
#endif  // MEMSHARE_STUB_ONLY
