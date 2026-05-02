# PHML Metal / MPS Backend

This directory contains the Apple Silicon GPU backend for `PHML::Data`. It is compiled only when the CMake option `PHML_WITH_MPS` is `ON` (the default on `arm64` macOS). Every public header in this directory is **metal-cpp-free**: `Tensor.hpp`, `Matrix.hpp`, and `Core.hpp` never pull in Metal headers, so the GPU layer is completely transparent to consumers on non-Apple platforms.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Build requirements](#2-build-requirements)
3. [Component reference](#3-component-reference)
   - [MPSAllocator](#31-mpsallocator)
   - [MetalContext](#32-metalcontext)
   - [MPSOps](#33-mpsops)
4. [Data flow: how a GPU op executes](#4-data-flow-how-a-gpu-op-executes)
5. [Unified memory and why it matters](#5-unified-memory-and-why-it-matters)
6. [Metal Shading Language kernels](#6-metal-shading-language-kernels)
7. [Dtype dispatch and CPU fallback](#7-dtype-dispatch-and-cpu-fallback)
8. [Thread safety](#8-thread-safety)
9. [Limitations and known constraints](#9-limitations-and-known-constraints)
10. [Extending the backend](#10-extending-the-backend)

---

## 1. Architecture overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Public API  (Tensor.hpp / Matrix.hpp)                           │
│                                                                  │
│  operator+, operator-, operator* (matmul & Hadamard), operator*  │
│  (scalar)  ──► #ifdef PHML_WITH_MPS  ──► is_mps() ?             │
│                    yes, float ──► mps::add_f32 / matmul_f32 …   │
│                    yes, other ──► warn_fallback_once + CPU loop  │
│                    no         ──► CPU loop                       │
└───────────────────────┬──────────────────────────────────────────┘
                        │  calls  (metal-cpp-free declarations)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  MPSOps.hpp / MPSOps.cpp   (PHML::Data::mps namespace)           │
│                                                                  │
│  matmul_f32  add_f32  sub_f32  mul_f32  scale_f32                │
│  warn_fallback_once                                              │
└───────────────────────┬──────────────────────────────────────────┘
                        │  uses
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  MetalContext  (singleton, owns all Metal state)                 │
│                                                                  │
│  MTL::Device*      ── system default GPU                         │
│  MTL::CommandQueue*── single queue for all dispatches            │
│  MTL::Library*     ── lazily compiled from embedded MSL source   │
│  pipeline cache    ── map<string, ComputePipelineState*>         │
│  buffer registry   ── map<void*, MTL::Buffer*>                   │
└───────────────────────┬──────────────────────────────────────────┘
                        │  device + registry
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  MPSAllocator  (implements Allocator interface from Core.hpp)    │
│                                                                  │
│  allocate(n)   ── newBuffer(n, StorageModeShared)                │
│                   register_buffer(contents, buf)                 │
│                   return contents  ◄── this IS the CPU pointer   │
│  deallocate(p) ── buffer_for(p) → release() + unregister         │
└──────────────────────────────────────────────────────────────────┘
```

The key design principle is **layered isolation**: Metal types never appear in public headers. Each layer communicates with the one below it through plain C++ pointers and the `Allocator` / `AllocatorRegistry` interfaces defined in `Core.hpp`.

---

## 2. Build requirements

| Requirement | Details |
|---|---|
| Platform | macOS on Apple Silicon (`arm64`) |
| CMake option | `-DPHML_WITH_MPS=ON` (default on arm64) |
| metal-cpp | Header-only, positioned at `third_party/metal-cpp/`. Download from [developer.apple.com/metal/cpp](https://developer.apple.com/metal/cpp/) |
| Linked frameworks | `Foundation`, `Metal`, `MetalPerformanceShaders`, `QuartzCore` |
| Compiler | Any C++20-capable Apple Clang (`xcode-select --install`) |

To build with MPS enabled:
```bash
cmake -S . -B build -DPHML_WITH_MPS=ON
cmake --build build -j
```

To build without (e.g. on Intel or CI):
```bash
cmake -S . -B build -DPHML_WITH_MPS=OFF
cmake --build build -j
```

When `PHML_WITH_MPS=OFF`, none of the backend headers or sources are compiled, no Metal frameworks are linked, and `Device::mps()` is still a valid enum value — but constructing `Storage` on it will throw `"No allocator registered for mps"`.

---

## 3. Component reference

### 3.1 MPSAllocator

**Files:** `MPSAllocator.hpp`, `src/Data/Backend/MPSAllocator.cpp`

Implements the `Allocator` interface (defined in `Core.hpp`) for `Device::mps()`. It is the only part of the backend wired into `Core.hpp` — via a forward declaration of `register_mps_allocator` and a call inside the `AllocatorRegistry` constructor.

**Lifecycle:**
- Registered automatically at program startup when `AllocatorRegistry::instance()` is first accessed (static singleton, called from `Storage` construction).
- `allocate(bytes)` — delegates to `MetalContext` to create an `MTL::Buffer` with `MTL::ResourceStorageModeShared`, then returns `buffer->contents()` (the CPU-accessible pointer). The `(ptr → buffer)` mapping is recorded in `MetalContext`'s buffer registry so ops code can recover the `MTL::Buffer*` from a raw pointer later.
- `deallocate(ptr, bytes)` — looks up the buffer via `MetalContext::buffer_for`, unregisters it, then calls `buffer->release()`.

**What it does NOT own:** The `MTL::Device*` now lives in `MetalContext`. This ensures the GPU device is shared between allocation and computation, and that the buffer registry is a single source of truth.

### 3.2 MetalContext

**Files:** `MetalContext.hpp`, `src/Data/Backend/MetalContext.cpp`

A process-wide singleton that owns every piece of Metal state. It is the only translation unit that defines `NS_PRIVATE_IMPLEMENTATION`, `MTL_PRIVATE_IMPLEMENTATION`, and `CA_PRIVATE_IMPLEMENTATION` — a metal-cpp requirement that exactly one TU provides the symbol definitions.

**Members:**

| Member | Type | Purpose |
|---|---|---|
| `device_` | `MTL::Device*` | System default Metal device, created at startup |
| `queue_` | `MTL::CommandQueue*` | Single command queue for all GPU submissions |
| `library_` | `MTL::Library*` | Compiled from `kElementwiseMSL` on first use |
| `pipelines_` | `map<string, PSO*>` | Per-kernel pipeline state cache |
| `buffers_` | `map<void*, Buffer*>` | Maps `contents()` pointers back to their `MTL::Buffer*` |
| `mutex_` | `std::mutex` | Protects all of the above |

**Key methods:**

- `instance()` — returns the singleton; first call constructs device + queue.
- `pipeline(fn_name)` — returns a cached `MTL::ComputePipelineState*`, compiling the library and building the PSO on the first call for a given kernel name. Thread-safe; uses the private `library_unlocked_()` helper to avoid a deadlock between `pipeline()` and `elementwise_library()` both trying to acquire `mutex_`.
- `buffer_for(ptr)` — resolves a raw CPU pointer (returned by `allocate`) back to its owning `MTL::Buffer*`. Throws if the pointer was not registered (i.e. was not allocated via `MPSAllocator` — this would be a programming error).
- `register_buffer` / `unregister_buffer` — called exclusively by `MPSAllocator`.

**Mutex design note:** All public methods that touch `library_`, `pipelines_`, or `buffers_` acquire `mutex_`. The `pipeline()` method calls `library_unlocked_()` (which assumes the lock is already held) to avoid nested lock acquisition.

### 3.3 MPSOps

**Files:** `MPSOps.hpp`, `src/Data/Backend/MPSOps.cpp`

The only backend header that `Tensor.hpp` and `Matrix.hpp` include (guarded by `#ifdef PHML_WITH_MPS`). It is **metal-cpp-free** — it contains only plain C++ declarations.

**Declared functions (namespace `PHML::Data::mps`):**

| Function | Operation | Kernel |
|---|---|---|
| `matmul_f32(A, B, C, M, K, N)` | C[M×N] = A[M×K] · B[K×N] | `matmul_f32` (2D grid) |
| `add_f32(a, b, out, n)` | out = a + b | `add_f32` (1D grid) |
| `sub_f32(a, b, out, n)` | out = a − b | `sub_f32` (1D grid) |
| `mul_f32(a, b, out, n)` | out = a ⊙ b (Hadamard) | `mul_f32` (1D grid) |
| `scale_f32(a, s, out, n)` | out = a · s | `scale_f32` (1D grid) |
| `warn_fallback_once(op, dtype)` | print once to stderr | — |

Each function follows the same dispatch pattern:
1. Resolve raw pointers to `MTL::Buffer*` via `MetalContext::buffer_for`.
2. Fetch (or lazily create) the `MTL::ComputePipelineState*` via `MetalContext::pipeline`.
3. Open a `MTL::CommandBuffer` + `MTL::ComputeCommandEncoder` via `encode_and_run`.
4. Bind buffers and inline constants with `setBuffer` / `setBytes`.
5. Dispatch threads, end encoding, `commit()`, `waitUntilCompleted()` (blocking).

---

## 4. Data flow: how a GPU op executes

Taking `Matrix<float>::operator*` on an MPS-resident matrix as a concrete example:

```
user code:
    Matrix<float> C = A * B;   // A, B on Device::mps()

Matrix.hpp  operator*(const Matrix<float>& other):
  1. shape check
  2. Matrix<float> result(rows_, other.cols_, device_)
        └─ Storage(bytes, Device::mps())
              └─ AllocatorRegistry::get(Device::mps())
                    └─ MPSAllocator::allocate(bytes)
                          └─ MetalContext::instance().device()
                                ->newBuffer(bytes, StorageModeShared)
                             MetalContext::register_buffer(ptr, buf)
                             return buf->contents()   ← CPU ptr stored in result
  3. #ifdef PHML_WITH_MPS, is_mps(), is float → mps::matmul_f32(A_ptr, B_ptr, C_ptr, M, K, N)

MPSOps.cpp  matmul_f32:
  4. ctx.buffer_for(A_ptr) → MTL::Buffer* bufA
     ctx.buffer_for(B_ptr) → MTL::Buffer* bufB
     ctx.buffer_for(C_ptr) → MTL::Buffer* bufC
  5. ctx.pipeline("matmul_f32") → ComputePipelineState* (cached after first call)
  6. encode_and_run:
       commandBuffer = queue->commandBuffer()
       encoder = commandBuffer->computeCommandEncoder()
       encoder->setComputePipelineState(pso)
       encoder->setBuffer(bufA, offset=0, index=0)
       encoder->setBuffer(bufB, offset=0, index=1)
       encoder->setBuffer(bufC, offset=0, index=2)
       encoder->setBytes({M,K,N}, 12, index=3)
       encoder->dispatchThreads({N, M, 1}, threadgroup={8, 8, 1})
       encoder->endEncoding()
       commandBuffer->commit()
       commandBuffer->waitUntilCompleted()   ← blocking sync
  7. return result   ← CPU can read result immediately via result.data_ptr<float>()
```

---

## 5. Unified memory and why it matters

On Apple Silicon, the CPU and GPU share the same physical DRAM. `MTL::ResourceStorageModeShared` creates a buffer whose memory is accessible to both processors simultaneously — `buffer->contents()` returns a CPU pointer that is literally the same physical memory the GPU reads and writes.

Consequences for this backend:

- **Zero-copy transfers.** `.to(Device::mps())` allocates a new `MTLBuffer` and copies the data once with `std::memcpy`. There is no separate DMA or staging buffer needed.
- **CPU fallback costs nothing.** When an op falls back to the CPU (unsupported dtype, non-contiguous layout), it runs the normal C++ loops directly on the `MTLBuffer`'s `contents()` pointer. The result is immediately GPU-visible. No explicit sync is needed.
- **Blocking sync is safe.** After `waitUntilCompleted()`, the GPU has finished writing to the shared buffer. The CPU can read the result immediately without a cache flush or barrier.
- **Apple Silicon only.** Intel Macs have discrete GPUs with separate VRAM. `StorageModeShared` does not exist for them; you would need `StorageModeManaged` (explicit sync) or `StorageModePrivate` (GPU-only, staging copy required). This backend does not support that path.

---

## 6. Metal Shading Language kernels

All kernels are compiled at runtime from the string constant `kElementwiseMSL` in `MetalContext.cpp`. No build-time shader compilation step is required. Compilation is triggered lazily on the first call to `MetalContext::pipeline()` and takes approximately 10–30 ms; subsequent calls hit the pipeline cache and are free.

### Element-wise kernels (1D)

Each kernel maps `tid` (thread ID) to one output element. The grid is `(n, 1, 1)` with threadgroup `(256, 1, 1)`. Apple Silicon supports **non-uniform threadgroup sizes** (`MTLGPUFamilyApple4`+), so the `dispatchThreads` call works correctly even when `n` is not a multiple of 256 — the hardware handles the boundary automatically.

Buffer layout (same for `add_f32`, `sub_f32`, `mul_f32`):

| Index | Binding | Direction |
|---|---|---|
| 0 | `device const float* a` | input |
| 1 | `device const float* b` | input |
| 2 | `device float* out` | output |
| 3 | `constant uint& n` | element count (inline via `setBytes`) |

`scale_f32` replaces buffer 1 with `constant float& s` (scalar passed via `setBytes`, not a buffer).

### GEMM kernel (2D)

`matmul_f32` dispatches a 2D grid of size `(N, M, 1)` with threadgroup `(8, 8, 1)`. Each thread computes one element `C[row][col]` by iterating over the inner dimension `K`.

```metal
kernel void matmul_f32(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device       float* C [[buffer(2)]],
                       constant     uint3& d [[buffer(3)]],  // {M, K, N}
                       uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y, col = gid.x;
    if (row >= d.x || col >= d.z) return;
    float acc = 0.0f;
    for (uint k = 0; k < d.y; ++k)
        acc += A[row * d.y + k] * B[k * d.z + col];
    C[row * d.z + col] = acc;
}
```

`d` is a packed `uint3` containing `{M, K, N}` passed inline with `setBytes`. This is a **naive GEMM** — one thread per output element, no shared-memory tiling. It is correct and GPU-accelerated but not peak-performance. See [§10](#10-extending-the-backend) for how to replace it with a tiled kernel.

---

## 7. Dtype dispatch and CPU fallback

GPU kernels are only implemented for `float` (float32). The dispatch sites in `Matrix.hpp` and `Tensor.hpp` use `if constexpr (std::is_same_v<T, float>)` to select the GPU path at compile time:

```cpp
#ifdef PHML_WITH_MPS
if (Base::is_mps() && is_contiguous()) {
    if constexpr (std::is_same_v<T, float>) {
        mps::matmul_f32(...);
        return result;
    } else {
        mps::warn_fallback_once("matrix::operator*", dtype_str(Base::dtype()));
    }
}
#endif
// CPU loop
```

When `T != float` (e.g. `double`, `int32_t`, `int64_t`), `warn_fallback_once` prints a one-time notice to `stderr`:

```
[PHML::MPS] 'matrix::operator*' has no GPU implementation for dtype 'float64';
running on CPU via shared-memory pointer (this notice is shown once).
```

The CPU loop then runs on the `MTLBuffer`'s `contents()` pointer — which is entirely valid on Apple Silicon (the buffer is readable and writable from the CPU at any time). The result is identical to what a CPU-only allocation would produce.

**Non-contiguous layouts** (e.g. after `transpose()` or `permute()`) also fall through to the CPU path silently, without a warning. This is intentional: it is a layout property, not a capability gap.

---

## 8. Thread safety

| Component | Guarantee |
|---|---|
| `MetalContext::instance()` | Initialization is thread-safe (C++11 static-local guarantee) |
| `MetalContext::pipeline()` | Thread-safe; `mutex_` held for cache lookup, compilation, and insertion |
| `MetalContext::buffer_for` / `register_buffer` / `unregister_buffer` | Thread-safe; `mutex_` held |
| `MPSAllocator::allocate` / `deallocate` | Thread-safe through `MetalContext` |
| `mps::warn_fallback_once` | Thread-safe; own static `mutex` + `unordered_set` |
| `encode_and_run` | Each call creates its own `MTL::CommandBuffer`; command buffers are not shared |

Note: `MetalContext` uses a single non-recursive `std::mutex`. The `pipeline()` method calls `library_unlocked_()` (which assumes the lock is already held) rather than re-entering `elementwise_library()` (which would deadlock). This pattern must be preserved when adding new methods.

---

## 9. Limitations and known constraints

| Limitation | Reason / Workaround |
|---|---|
| Apple Silicon only | `StorageModeShared` does not exist on discrete GPUs. Intel Mac / CUDA support would require managed/private buffers and blit-encoder copies. |
| `float32` only on GPU | Apple Silicon shaders support `half` and `float`; not `double`. `double` ops use the CPU fallback. |
| Non-contiguous tensors always use CPU | The GPU kernels assume row-major contiguous layout. Calling `.contiguous()` materialises a copy that can then be GPU-dispatched. |
| Blocking sync (`waitUntilCompleted`) | Each op is an independent GPU submission. Chaining `a + b + c` submits two separate command buffers. Async batching is not yet implemented. |
| Naive GEMM (no tiling) | `matmul_f32` is O(M·K·N) with one thread per output element and no shared-memory reuse. Throughput is memory-bandwidth limited for large matrices. |
| No `half` / `bfloat16` | Not yet wired; would need a new `DType::Float16` entry and corresponding kernel variants. |
| Single `MTL::CommandQueue` | Fine for sequential use. Concurrent GPU submissions from multiple threads would benefit from a queue-per-thread or a submission pool. |

---

## 10. Extending the backend

### Adding a new kernel

1. Write the MSL function in the `kElementwiseMSL` string inside `MetalContext.cpp`.
2. Declare the C++ wrapper in `MPSOps.hpp` (metal-cpp-free signature).
3. Implement it in `MPSOps.cpp` following the `encode_and_run` pattern.
4. Add the dispatch block (with `if constexpr` dtype guard and fallback warning) to the relevant operator in `Matrix.hpp` or `Tensor.hpp`.

### Replacing the naive GEMM with a tiled kernel

The `matmul_f32` kernel in `kElementwiseMSL` can be replaced with a threadgroup-shared-memory tiled GEMM without any changes to the dispatch layer. A standard 16×16 tile reduces global memory accesses by ~16× and is typically 5–10× faster for large matrices on M-series GPUs. The function signature, buffer layout, and grid dimensions remain the same; only the kernel body changes.

### Adding `float16` support

1. Add `DType::Float16` to `Core.hpp` with `dtype_size` = 2 and a string name.
2. Add `half` kernel variants to `kElementwiseMSL` (e.g. `add_f16`, `matmul_f16`).
3. Add `mps::add_f16`, `mps::matmul_f16` declarations and implementations.
4. Add `if constexpr (std::is_same_v<T, _Float16>)` branches in the operator dispatch blocks.

### Supporting discrete GPUs (Intel Mac / external GPU)

Replace `StorageModeShared` with `StorageModeManaged` (CPU+GPU accessible, explicit sync) or `StorageModePrivate` (GPU-only, requires blit encoder for transfers). The `MPSAllocator` and `MetalContext` are the only files that need changing; the rest of the backend is pointer-agnostic.
