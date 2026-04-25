# Data Structures — `PHML` ML Framework

This document is the reference for all data structures in the **PHML** framework. Each section covers one type's design, memory model, and public API.

---

## Table of Contents

- [Core Module](#core-module)
  1. [Requirements](#requirements)
  2. [Architecture Overview](#architecture-overview)
  3. [Components](#components)
     - [Device](#1-device)
     - [DType](#2-dtype)
     - [Allocator & CPUAllocator](#3-allocator--cpuallocator)
     - [AllocatorRegistry](#4-allocatorregistry)
     - [Storage](#5-storage)
     - [Core\<Derived\>](#6-corederived)
  4. [Extending Core — Writing a New Data Structure](#extending-core--writing-a-new-data-structure)
  5. [Extending the Allocator — Adding CUDA Support](#extending-the-allocator--adding-cuda-support)
  6. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
  7. [Planned Extensions](#planned-extensions)
  8. [API Reference — Core](#api-reference)
- [Tensor\<T\>](#tensort)
  1. [Overview](#tensor-overview)
  2. [Memory Layout](#memory-layout)
  3. [Element Addressing](#element-addressing)
  4. [View Semantics](#view-semantics)
  5. [Contiguous vs Non-Contiguous](#contiguous-vs-non-contiguous)
  6. [Operations](#operations)
  7. [Factories](#factories)
  8. [API Reference — Tensor](#api-reference--tensor)

---

# Core Module

`Core.hpp` is the single-header foundation of the **PHML** framework.  
Every data structure in the framework — `Matrix<T>`, `Tensor<T>`, and any future type — inherits from the CRTP base class `Core<Derived>` defined here.

The module is deliberately minimal: it owns **no math, no ops, no I/O**. Its only job is to standardise how data structures describe themselves (device, dtype) and how they allocate and share memory.

---

## Requirements

| Requirement | Version |
|---|---|
| C++ Standard | C++17 or later |
| Compiler | GCC ≥ 9, Clang ≥ 10, MSVC ≥ 19.29 |
| Dependencies | None (standard library only) |

```cpp
#include "core.hpp"   // everything is in this single header
```

---

## Architecture Overview

The module is composed of **five layers**, each with a single responsibility. They are stacked bottom-up:

```
┌─────────────────────────────────────────────────┐
│              Core<Derived>  (CRTP base)          │  ← layer 5
│   owns dtype_, device_, shared_ptr<Storage>     │
├─────────────────────────────────────────────────┤
│                   Storage                        │  ← layer 4
│   ref-counted raw buffer, device-aware           │
├─────────────────────────────────────────────────┤
│       AllocatorRegistry  (singleton)             │  ← layer 3b
│   maps Device → Allocator                       │
├─────────────────────────────────────────────────┤
│     Allocator  (interface)  /  CPUAllocator      │  ← layer 3a
│   pluggable alloc / dealloc per device           │
├─────────────────────────────────────────────────┤
│         Device          DType                    │  ← layers 1 & 2
│   where data lives    what scalars are           │
└─────────────────────────────────────────────────┘
```

All types live inside the `mlf` namespace.

---

## Components

### 1. `Device`

Describes **where data lives**: which hardware backend and which index of that backend.

```cpp
struct Device {
    DeviceType  type;   // CPU, CUDA, …
    int         index;  // e.g. 0 for cuda:0, 1 for cuda:1
};
```

#### Factory helpers

```cpp
Device cpu_dev  = Device::cpu();       // { CPU,  0 }
Device cuda0    = Device::cuda(0);     // { CUDA, 0 }
Device cuda1    = Device::cuda(1);     // { CUDA, 1 }
```

#### Member functions

| Function | Return | Description |
|---|---|---|
| `is_cpu()` | `bool` | `true` when `type == DeviceType::CPU` |
| `is_cuda()` | `bool` | `true` when `type == DeviceType::CUDA` |
| `str()` | `std::string` | Human-readable: `"cpu"`, `"cuda:0"`, … |
| `operator==` | `bool` | Equality on `(type, index)` |

#### `DeviceType` enum

```cpp
enum class DeviceType : uint8_t {
    CPU  = 0,
    CUDA = 1,
    // Future: METAL, VULKAN, ...
};
```

Stored as `uint8_t` to keep `Device` a trivially small struct (5 bytes padded to 8).

---

### 2. `DType`

A **runtime tag** for the scalar element type of a data structure. Keeping it as a runtime value (rather than encoding it only in the C++ template parameter) lets generic code — serialisation, op dispatch, printing — branch on element type without template specialisations.

```cpp
enum class DType : uint8_t {
    Float32 = 0,
    Float64,
    Int32,
    Int64,
    Bool,
    // Future: Float16, BFloat16, ...
};
```

#### Free functions

```cpp
// Returns byte size of one element
std::size_t dtype_size(DType dt);

// Returns human-readable name: "float32", "float64", ...
std::string dtype_str(DType dt);
```

#### Example

```cpp
DType dt = DType::Float32;

std::cout << dtype_str(dt);    // "float32"
std::cout << dtype_size(dt);   // 4
```

---

### 3. `Allocator` & `CPUAllocator`

`Allocator` is a **pure-virtual interface** that decouples memory allocation from the rest of the system. Swapping in a CUDA or custom arena allocator requires no changes to `Core`, `Storage`, or any data structure.

```cpp
struct Allocator {
    virtual void*  allocate  (std::size_t bytes)             = 0;
    virtual void   deallocate(void* ptr, std::size_t bytes)  = 0;
    virtual Device device    () const                        = 0;
};
```

`CPUAllocator` is the built-in implementation, backed by `std::malloc` / `std::free`. It is registered automatically for `Device::cpu()` at program startup.

```cpp
struct CPUAllocator final : Allocator {
    void*  allocate  (std::size_t bytes) override;   // malloc
    void   deallocate(void* ptr, std::size_t) override; // free
    Device device    () const override;              // Device::cpu()
};
```

To add a custom allocator for any device:

```cpp
struct MyArenaAllocator : mlf::Allocator {
    void* allocate(std::size_t bytes) override { /* ... */ }
    void  deallocate(void* ptr, std::size_t bytes) override { /* ... */ }
    mlf::Device device() const override { return mlf::Device::cpu(); }
};

// Register before any data structure is created:
mlf::AllocatorRegistry::instance().register_allocator(
    mlf::Device::cpu(),
    std::make_shared<MyArenaAllocator>()
);
```

---

### 4. `AllocatorRegistry`

A **Meyer's singleton** that maps each `Device` to exactly one `Allocator`. `Storage` calls into this registry every time it needs to allocate or free memory.

```cpp
class AllocatorRegistry {
public:
    static AllocatorRegistry& instance();

    void       register_allocator(Device dev, std::shared_ptr<Allocator> alloc);
    Allocator& get               (Device dev);
};
```

The registry is initialised with a `CPUAllocator` for `Device::cpu()` before `main()` runs. Registering a new allocator for a device that already has one silently replaces it (last-write-wins).

**Thread safety note:** `register_allocator` is not thread-safe. Register all allocators at program startup, before any data structure is constructed.

---

### 5. `Storage`

A **non-copyable, movable, reference-counted raw buffer** that owns a contiguous region of memory on a specific device.

```cpp
class Storage {
public:
    Storage(std::size_t bytes, Device dev);

    void*       data();
    const void* data() const;
    std::size_t size()   const;  // byte count
    Device      device() const;
};
```

`Storage` objects are never created directly by user code. They are created by `Core::init_storage()` and held inside a `std::shared_ptr<Storage>`. This means multiple data structures (e.g. a matrix and its transposed view) can safely point at the same backing buffer — when the last owner goes out of scope, the buffer is freed automatically.

```
Matrix A  ──┐
             ├──► shared_ptr<Storage>  ──► raw buffer
Matrix A.T ─┘
```

Modifying elements through `A` is immediately visible through `A.T` because they share the same `Storage`.

---

### 6. `Core<Derived>`

The **CRTP base class** that every data structure inherits from. It aggregates the layers above and exposes a uniform interface for device/dtype queries, storage management, and device transfer.

```cpp
template <typename Derived>
class Core { ... };
```

#### Why CRTP instead of a virtual base?

Virtual dispatch costs a pointer dereference on every method call — unacceptable on hot ML inner loops. CRTP provides identical polymorphic structure with **zero runtime overhead**: `self()` is a compile-time `static_cast`, resolved entirely by the compiler.

```cpp
// Pattern used by all data structures:
class Matrix : public Core<Matrix> { ... };
class Tensor : public Core<Tensor> { ... };
```

#### Constructor

```cpp
// Default — leaves dtype=Float32, device=cpu
Core();

// Explicit — called by derived classes
Core(DType dtype, Device device);
```

Derived classes must call the explicit constructor from their own constructor:

```cpp
Matrix(std::size_t rows, std::size_t cols, Device dev = Device::cpu())
    : Core<Matrix>(DType::Float32, dev)  // ← required
    , rows_(rows), cols_(cols)
{
    init_storage(rows * cols * sizeof(float));
}
```

#### Device & dtype queries

```cpp
DType  dtype()   const;  // e.g. DType::Float32
Device device()  const;  // e.g. Device::cpu()
bool   is_cpu()  const;
bool   is_cuda() const;
```

#### Storage management

```cpp
// Called once by the derived class constructor after shape is known
void init_storage(std::size_t bytes);

// Shared-pointer access — for views, slices, etc.
std::shared_ptr<Storage> storage() const;
bool                     has_storage() const;

// Typed raw pointer — primary way to read/write elements
template <typename T> T*       data_ptr();
template <typename T> const T* data_ptr() const;
```

`data_ptr<T>()` throws `std::runtime_error` if `init_storage()` has not been called yet.

#### CRTP self() helper

```cpp
Derived&       self();        // static_cast<Derived&>(*this)
const Derived& self() const;
```

Used internally by `Core` to call methods on the derived class without virtual dispatch. Derived classes rarely need to call this directly.

#### Device transfer hook

```cpp
Derived to(Device target) const;
```

The base implementation is a **guard**: it returns `*this` if `target == device_`, and throws `std::runtime_error` for any other device. Derived classes are expected to override this method with a real copy/transfer implementation.

```cpp
// In Matrix<T>:
Matrix<T> to(Device target) const {
    if (Base::device_ == target) return *this;
    // ... allocate dst on target, memcpy / cudaMemcpy ...
}
```

#### Debug metadata

```cpp
std::string meta_str() const;
// Returns: "dtype=float32 device=cpu"
```

#### Protected members available to derived classes

```cpp
protected:
    DType                    dtype_;    // = DType::Float32
    Device                   device_;   // = Device::cpu()
    std::shared_ptr<Storage> storage_;  // = nullptr until init_storage()

    // Reserved for future use (uncomment as needed):
    // bool        requires_grad_ = false;
    // GradFn      grad_fn_       = nullptr;
    // std::string name_          = "";
```

---

## Extending Core — Writing a New Data Structure

The following checklist covers everything required to add a new type (e.g. `Tensor<T>`):

**1. Inherit from `Core<YourType>`**

```cpp
template <typename T>
class Tensor : public mlf::Core<Tensor<T>> {
    using Base = mlf::Core<Tensor<T>>;
    // ...
};
```

**2. Call the `Core` constructor with the correct dtype and device**

```cpp
Tensor(std::vector<std::size_t> shape, mlf::Device dev = mlf::Device::cpu())
    : Base(dtype_of<T>(), dev)
    , shape_(shape)
{
    std::size_t numel = 1;
    for (auto d : shape) numel *= d;
    Base::init_storage(numel * sizeof(T));
}
```

**3. Use `Base::data_ptr<T>()` to access elements**

```cpp
T& operator[](std::size_t flat_idx) {
    return Base::template data_ptr<T>()[flat_idx];
}
```

**4. Override `to(Device)` for device transfers**

```cpp
Tensor<T> to(mlf::Device target) const {
    if (Base::device_ == target) return *this;
    Tensor<T> dst(shape_, target);
    std::memcpy(dst.template data_ptr<T>(),
                Base::template data_ptr<T>(),
                numel() * sizeof(T));
    return dst;
}
```

**5. Zero-copy views share `storage_`**

```cpp
Tensor<T> slice(/* range params */) const {
    Tensor<T> view;
    view.dtype_   = Base::dtype_;
    view.device_  = Base::device_;
    view.storage_ = Base::storage_;  // shared — no alloc
    // adjust shape/strides for the slice
    return view;
}
```

---

## Extending the Allocator — Adding CUDA Support

```cpp
#include "core.hpp"
#include <cuda_runtime.h>

struct CUDAAllocator final : mlf::Allocator {
    explicit CUDAAllocator(int device_index) : idx_(device_index) {}

    void* allocate(std::size_t bytes) override {
        void* ptr = nullptr;
        cudaSetDevice(idx_);
        cudaMalloc(&ptr, bytes);
        return ptr;
    }

    void deallocate(void* ptr, std::size_t /*bytes*/) override {
        cudaFree(ptr);
    }

    mlf::Device device() const override {
        return mlf::Device::cuda(idx_);
    }

private:
    int idx_;
};

// Call once at program startup, before creating any data structures:
void init_cuda_allocators(int num_gpus) {
    for (int i = 0; i < num_gpus; ++i) {
        mlf::AllocatorRegistry::instance().register_allocator(
            mlf::Device::cuda(i),
            std::make_shared<CUDAAllocator>(i)
        );
    }
}
```

After registration, creating any data structure on a CUDA device will automatically use this allocator:

```cpp
init_cuda_allocators(1);
Matrix<float> gpu_mat(1024, 1024, mlf::Device::cuda(0));  // allocated on GPU
```

---

## Design Decisions & Trade-offs

### CRTP vs virtual inheritance

CRTP was chosen over a `virtual` base class for two reasons. First, there is **zero runtime cost**: no vtable pointer per object, no indirect call on every access. Second, it enables **static interface checking** — if a derived class fails to implement a method that `Core` expects, the error is caught at compile time, not at runtime.

The trade-off is that heterogeneous collections (`std::vector<Core*>`) are not possible. If runtime polymorphism is needed later, a thin wrapper with a virtual interface can be placed on top of the CRTP hierarchy.

### Shared `Storage` for zero-copy views

`Storage` is held behind a `shared_ptr` rather than being embedded by value. This means views, transposes, and slices all share one backing buffer with no extra allocation. The cost is an extra heap allocation for the control block and a slightly larger per-object footprint (one pointer). For ML workloads this trade-off heavily favours shared storage.

### Runtime `DType` alongside compile-time `T`

`Matrix<float>` encodes its element type both as the C++ template parameter `T` and as the runtime `DType::Float32` tag. The compile-time type is used for type-safe element access (`data_ptr<float>()`). The runtime tag is used for generic code (printing, serialisation, op dispatch tables) that cannot be specialised over every possible `T`.

### `AllocatorRegistry` as a Meyer's singleton

The registry must be available before `main()` (to register `CPUAllocator`) and must outlive all data structures (to deallocate their storage in destructors). A Meyer's singleton satisfies both constraints with no dynamic initialisation order issues. The trade-off is that it is a global — test isolation requires care if tests register non-default allocators.

### `Device` as a value type (not a pointer or enum)

Storing `Device` by value (8 bytes) makes it trivially copyable and cheaply comparable. Representing the device index alongside the type (rather than just the type) means the design is multi-GPU-ready without any interface change.

---

## Planned Extensions

The following features are intentionally left as stubs or comments in the current implementation. They will be added without breaking backward compatibility:

| Feature | Location | Notes |
|---|---|---|
| Autograd metadata | `Core` protected fields | `requires_grad_`, `grad_fn_` |
| Float16 / BFloat16 | `DType` enum | Requires half-precision support in math ops |
| Metal / Vulkan backends | `DeviceType` enum + new `Allocator` | Mirror the `CUDAAllocator` pattern |
| Named tensors | `Core` protected fields | `std::string name_` |
| Memory pinning | `CPUAllocator` subclass | `posix_memalign` / `cudaMallocHost` |
| Thread-safe registry | `AllocatorRegistry` | `std::shared_mutex` around `map_` |

---

## API Reference

### Free functions (`core.hpp`)

| Signature | Description |
|---|---|
| `std::size_t dtype_size(DType)` | Byte width of one element |
| `std::string dtype_str(DType)` | Human-readable dtype name |

### `Device`

| Member | Description |
|---|---|
| `Device::cpu()` | Factory for CPU device |
| `Device::cuda(int idx=0)` | Factory for CUDA device |
| `bool is_cpu() const` | True if CPU |
| `bool is_cuda() const` | True if CUDA |
| `std::string str() const` | `"cpu"` or `"cuda:N"` |
| `bool operator==(const Device&) const` | Equality |

### `Allocator` (interface)

| Member | Description |
|---|---|
| `void* allocate(size_t bytes)` | Allocate `bytes` on the device |
| `void deallocate(void*, size_t)` | Release previously allocated pointer |
| `Device device() const` | Device this allocator serves |

### `AllocatorRegistry`

| Member | Description |
|---|---|
| `static AllocatorRegistry& instance()` | Singleton accessor |
| `void register_allocator(Device, shared_ptr<Allocator>)` | Register or replace |
| `Allocator& get(Device)` | Get allocator; throws if not registered |

### `Storage`

| Member | Description |
|---|---|
| `Storage(size_t bytes, Device)` | Allocate via registry |
| `void* data()` | Raw pointer to buffer |
| `size_t size() const` | Byte count |
| `Device device() const` | Owning device |

### `Core<Derived>`

| Member | Description |
|---|---|
| `DType dtype() const` | Runtime dtype tag |
| `Device device() const` | Owning device |
| `bool is_cpu() const` | True if on CPU |
| `bool is_cuda() const` | True if on CUDA |
| `void init_storage(size_t bytes)` | Allocate backing buffer (call once) |
| `shared_ptr<Storage> storage() const` | Access shared storage |
| `bool has_storage() const` | True after `init_storage()` |
| `T* data_ptr<T>()` | Typed raw pointer |
| `const T* data_ptr<T>() const` | Const typed raw pointer |
| `Derived& self()` | CRTP cast to derived |
| `Derived to(Device) const` | Device transfer (override in derived) |
| `std::string meta_str() const` | Debug string |

---

# Tensor\<T\>

**Header:** `include/PHML/Data/Tensor.hpp`

```cpp
#include "PHML/Data/Tensor.hpp"

using namespace PHML::Data;
```

---

## Tensor Overview

`Tensor<T>` is an N-dimensional array of scalar type `T`. It inherits from `Core<Tensor<T>>` for device/dtype bookkeeping and shared storage management, and adds its own shape, strides, and indexing on top.

Supported scalar types and their runtime `DType` tags:

| C++ type   | `DType`          |
|------------|-----------------|
| `float`    | `DType::Float32` |
| `double`   | `DType::Float64` |
| `int32_t`  | `DType::Int32`   |
| `int64_t`  | `DType::Int64`   |
| `bool`     | `DType::Bool`    |

The backing buffer is a single flat array allocated by `Core::init_storage()` through the `AllocatorRegistry`. All element access goes through an offset computed from the tensor's shape and strides.

---

## Memory Layout

A `Tensor<T>` stores three pieces of metadata alongside the flat buffer:

```
shape_   : [d0, d1, ..., dN-1]   ← size along each dimension
strides_ : [s0, s1, ..., sN-1]   ← elements to skip per step on each axis
numel_   : d0 * d1 * ... * dN-1  ← total element count
```

The flat buffer always holds `numel_ * sizeof(T)` bytes. The shape and strides together define how logical multi-dimensional indices map onto that flat buffer.

### Default (row-major / C-order) strides

When a tensor is first constructed, strides are set to the standard row-major layout by `default_strides()`:

```
shape   = [d0, d1, d2]
strides = [d1*d2,  d2,  1]
```

More generally, `strides[i] = strides[i+1] * shape[i+1]`, filled right-to-left with `strides[N-1] = 1`.

**Example — a `[3, 4, 5]` tensor:**

```
strides = [20, 5, 1]
```

Consecutive elements along the last axis are adjacent in memory (stride 1). Moving one step along the first axis skips 20 elements (one full 4×5 slice).

This layout is identical to a C multidimensional array and is optimal for row-major traversal.

### Memory diagram

```
shape = [2, 3]    strides = [3, 1]

Logical:          Physical buffer:
  [0,0] [0,1] [0,2]     [a, b, c, d, e, f]
  [1,0] [1,1] [1,2]      ↑
                         offset(0,0) = 0*3 + 0*1 = 0
                         offset(0,2) = 0*3 + 2*1 = 2
                         offset(1,0) = 1*3 + 0*1 = 3
                         offset(1,2) = 1*3 + 2*1 = 5
```

---

## Element Addressing

Every element access reduces to a **dot product** of the index vector and the strides vector:

```
offset(i0, i1, ..., iN-1) = i0*s0 + i1*s1 + ... + iN-1*sN-1
```

Two functions implement this:

### `offset_of` (bounds-checked)

Called by `operator()` and `at()`. Verifies that the number of supplied indices matches `ndim()` and that each index is within bounds, then computes the dot product:

```cpp
// operator()(i, j, k) compiles to:
std::size_t arr[] = { i, j, k };
return data_ptr<T>()[offset_of(arr, 3)];
```

Throws `std::invalid_argument` on rank mismatch, `std::out_of_range` on out-of-bounds index.

### `at_offset` (unchecked)

Used internally by `print_dim`, `binary_op`, and `walk_indices`. Takes a pre-built index vector and computes the dot product without any range checks — safe because the callers already control the index range.

---

## View Semantics

`reshape`, `permute`, and `transpose` all return a **view**: a new `Tensor` object that shares the same `Storage` as the original but carries different shape and/or strides. No data is copied.

```
Original tensor A           View B = A.transpose(0, 1)
┌──────────────────┐        ┌──────────────────┐
│ shape_  = [3, 4] │        │ shape_  = [4, 3] │
│ strides_= [4, 1] │        │ strides_= [1, 4] │
│ storage_─────────┼──┐     │ storage_─────────┼──┐
└──────────────────┘  │     └──────────────────┘  │
                      └──────────► shared buffer   │
                                  [a,b,c,d,e,f,…]◄─┘
```

Writing through `B` immediately affects `A` because they share the same backing buffer.

### `reshape`

Changes shape and recomputes default strides. Requires the tensor to already be contiguous (strides must match the default row-major layout); throws otherwise. The new shape must have the same `numel_`.

```cpp
Tensor<float> t({6}, 0.0f);         // shape [6]
auto mat = t.reshape({2, 3});       // shape [2,3], strides [3,1]
```

### `permute`

Reorders both `shape_` and `strides_` in lockstep according to the supplied axis permutation. The buffer is untouched; only the metadata changes.

```cpp
// [3, 4, 5] tensor with strides [20, 5, 1]
auto p = t.permute({2, 0, 1});
// shape   = [5, 3, 4]
// strides = [1, 20, 5]   ← same values, reordered
```

### `transpose`

Convenience wrapper around `permute` that swaps exactly two axes.

```cpp
auto t2 = t.transpose(0, 1);   // swap axes 0 and 1
```

---

## Contiguous vs Non-Contiguous

A tensor is **contiguous** when its strides equal `default_strides(shape_)` — that is, when consecutive logical elements are also consecutive in the physical buffer. `is_contiguous()` checks this by comparing `strides_` to the default.

After `permute` or `transpose`, the strides no longer match the default layout, so the tensor is non-contiguous.

### Why it matters

**1. Correctness of raw iteration**

`begin()` and `end()` return raw pointers into the flat buffer. Iterating `[begin(), end())` visits elements in physical storage order, which matches logical row-major order only when the tensor is contiguous. On a non-contiguous tensor this iteration is logically incorrect.

**2. Performance**

Contiguous tensors are traversed with a simple pointer loop:

```cpp
const T* a = data_ptr<T>();
for (std::size_t i = 0; i < numel_; ++i)
    result[i] = a[i] * scalar;
```

Non-contiguous tensors use `walk_indices`, which maintains a full index vector and advances it digit-by-digit on every step. This adds per-element overhead and, more critically, accesses memory in a non-sequential pattern — causing CPU cache misses on large tensors.

```
Contiguous [4, 4] read pattern:   Non-contiguous (transposed) read:
→ → → → → → → →                   ↓ ↓ ↓ ↓
sequential, prefetch-friendly      column-major, cache-thrashing
```

### Making a tensor contiguous

`.contiguous()` materialises a non-contiguous view into a new tensor with default strides. The copy is a sequential write — cache-friendly — and needs to happen only once before a series of operations:

```cpp
auto view = t.transpose(0, 1);          // zero-copy, non-contiguous
auto c    = view.contiguous();          // one sequential copy
auto result = c + other;                // fast contiguous path
```

`.to(device)` always produces a contiguous copy regardless of the source layout.

---

## Operations

All element-wise arithmetic operators (`+`, `-`, `*`) check `is_contiguous()` and branch:

- **Both contiguous:** pointer loop over the flat buffer — O(numel) with no overhead.
- **Either non-contiguous:** `walk_indices` with stride-based addressing — correct for any layout but slower.

Scalar multiplication (`tensor * scalar` and `scalar * tensor`) follows the same pattern.

The `shape_check` helper is called at the start of every binary operation and throws `std::invalid_argument` on shape mismatch.

---

## Factories

Static factory methods on `Tensor<T>`:

| Factory | Description |
|---------|-------------|
| `Tensor<T>::zeros(shape, dev)` | All elements set to `T{0}` |
| `Tensor<T>::ones(shape, dev)` | All elements set to `T{1}` |
| `Tensor<T>::full(shape, val, dev)` | All elements set to `val` |
| `Tensor<T>::random(shape, low, high, dev)` | Uniform random in `[low, high]`; uses `std::mt19937_64` |

`TensorFactory` provides dtype-erased versions of `zeros` and `random` that return an `AnyTensor` (`std::variant` of all supported concrete types), useful when the scalar type is only known at runtime:

```cpp
AnyTensor t = TensorFactory::zeros({3, 4}, DType::Float32);
```

---

## API Reference — Tensor

### Constructors

| Signature | Description |
|-----------|-------------|
| `Tensor()` | Default — empty tensor, no storage |
| `Tensor(shape, dev)` | Allocate zero-initialised |
| `Tensor(shape, default_val, dev)` | Allocate filled with `default_val` |
| `Tensor(shape, initializer_list<T>, dev)` | Allocate from flat value list (size must match numel) |

### Shape & layout queries

| Member | Description |
|--------|-------------|
| `size_t ndim() const` | Number of dimensions |
| `const vector<size_t>& shape() const` | Full shape vector |
| `size_t shape(size_t i) const` | Size along axis `i` (bounds-checked) |
| `const vector<size_t>& strides() const` | Strides vector |
| `size_t numel() const` | Total element count |
| `bool is_contiguous() const` | True if strides match row-major default |

### Element access

| Member | Description |
|--------|-------------|
| `T& operator()(Idxs... idxs)` | Variadic, compile-time rank, bounds-checked |
| `const T& operator()(Idxs...) const` | Const overload |
| `T& at(initializer_list<size_t>)` | Runtime-rank, bounds-checked |
| `const T& at(...) const` | Const overload |

### Views (zero-copy)

| Member | Description |
|--------|-------------|
| `Tensor reshape(vector<size_t>) const` | New shape, same numel; requires contiguous |
| `Tensor permute(vector<size_t> axes) const` | Reorder dimensions |
| `Tensor transpose(size_t a, size_t b) const` | Swap two axes |
| `Tensor contiguous() const` | Materialise into a fresh contiguous tensor; no-op if already contiguous |

### Device transfer

| Member | Description |
|--------|-------------|
| `Tensor to(Device) const` | Copy to target device; no-op if already there |

### Arithmetic

| Member | Description |
|--------|-------------|
| `Tensor operator+(const Tensor&) const` | Element-wise addition |
| `Tensor operator-(const Tensor&) const` | Element-wise subtraction |
| `Tensor operator*(const Tensor&) const` | Element-wise (Hadamard) product |
| `Tensor operator*(T scalar) const` | Scalar multiplication |
| `friend Tensor operator*(T, const Tensor&)` | Scalar multiplication (reversed) |

### Iterators

| Member | Description |
|--------|-------------|
| `T* begin()` / `T* end()` | Raw pointers into the flat buffer |
| `const T* begin() const` / `const T* end() const` | Const overloads |

Valid for logical traversal **only when `is_contiguous()` is true**.

### Printing

| Member | Description |
|--------|-------------|
| `void print(ostream& os) const` | Nested-bracket layout with shape header |
| `friend ostream& operator<<(ostream&, const Tensor&)` | Stream operator |