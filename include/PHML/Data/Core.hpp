#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace PHML::Data {  // ml framework

// ============================================================
// 1. DEVICE
//    Represents where data lives. Extensible to multi-GPU.
// ============================================================

enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    MPS = 2,
    // METAL, VULKAN, ...
};

struct Device {
    DeviceType  type;
    int         index;  // e.g. cuda:0, cuda:1

    // Factory helpers
    static Device cpu()          { return {DeviceType::CPU,  0}; }
    static Device cuda(int idx=0){ return {DeviceType::CUDA, idx}; }
    static Device mps(int idx=0) { return {DeviceType::MPS, idx}; }

    bool is_cpu()  const { return type == DeviceType::CPU;  }
    bool is_cuda() const { return type == DeviceType::CUDA; }
    bool is_mps() const { return type == DeviceType::MPS; }

    bool operator==(const Device& o) const {
        return type == o.type && index == o.index;
    }

    std::string str() const {
        if (is_cpu()) return "cpu";
        if (is_cuda()) return "cuda:" + std::to_string(index);
        if (is_mps())  return "mps:" + std::to_string(index);
        return "unknown";
    }
};


// ============================================================
// 2. DTYPE
//    Scalar element type, stored as a runtime tag so tensors
//    and matrices can be generic without full template arity.
// ============================================================

enum class DType : uint8_t {
    Float32 = 0,
    Float64,
    Int32,
    Int64,
    Bool,
    // Future: Float16, BFloat16, ...
};

inline std::size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::Bool:    return 1;
        default: throw std::runtime_error("Unknown DType");
    }
}

inline std::string dtype_str(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        case DType::Bool:    return "bool";
        default: return "unknown";
    }
}


// ============================================================
// 3. MEMORY ALLOCATOR INTERFACE
//    Swap in a CUDA allocator later with zero changes to Core.
// ============================================================

struct Allocator {
    virtual ~Allocator() = default;

    virtual void* allocate(std::size_t bytes) = 0;
    virtual void  deallocate(void* ptr, std::size_t bytes) = 0;
    virtual Device device() const = 0;
};

// --- CPU allocator (default) --------------------------------

struct CPUAllocator final : Allocator {
    void* allocate(std::size_t bytes) override {
        void* p = std::malloc(bytes);
        if (!p) throw std::bad_alloc();
        return p;
    }
    void deallocate(void* ptr, std::size_t /*bytes*/) override {
        std::free(ptr);
    }
    Device device() const override { return Device::cpu(); }
};

// --- Allocator registry: one allocator per device -----------

#ifdef PHML_WITH_MPS
class AllocatorRegistry;
void register_mps_allocator(AllocatorRegistry&);
#endif

class AllocatorRegistry {
public:
    static AllocatorRegistry& instance() {
        static AllocatorRegistry reg;
        return reg;
    }

    void register_allocator(Device dev, std::shared_ptr<Allocator> alloc) {
        map_[key(dev)] = std::move(alloc);
    }

    Allocator& get(Device dev) {
        auto it = map_.find(key(dev));
        if (it == map_.end())
            throw std::runtime_error("No allocator registered for " + dev.str());
        return *it->second;
    }

private:
    AllocatorRegistry() {
        register_allocator(Device::cpu(),
                           std::make_shared<CPUAllocator>());
#ifdef PHML_WITH_MPS
        register_mps_allocator(*this);
#endif
    }

    static std::string key(Device d) { return d.str(); }
    std::unordered_map<std::string, std::shared_ptr<Allocator>> map_;
};


// ============================================================
// 4. STORAGE
//    A reference-counted raw buffer that knows its device.
//    Shared between views / slices of the same data.
// ============================================================

class Storage {
public:
    Storage(std::size_t bytes, Device dev)
        : bytes_(bytes), device_(dev)
    {
        Allocator& alloc = AllocatorRegistry::instance().get(dev);
        ptr_ = alloc.allocate(bytes);
    }

    ~Storage() {
        if (ptr_) {
            try {
                AllocatorRegistry::instance().get(device_)
                    .deallocate(ptr_, bytes_);
            } catch (...) { /* best-effort */ }
        }
    }

    // Non-copyable, movable
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_), device_(o.device_) {
        o.ptr_ = nullptr;
    }

    void*       data()         { return ptr_;     }
    const void* data()   const { return ptr_;     }
    std::size_t size()   const { return bytes_;   }
    Device      device() const { return device_;  }

private:
    void*       ptr_;
    std::size_t bytes_;
    Device      device_;
};


// ============================================================
// 5. CORE  (CRTP base)
//
//    Every data structure (Matrix<T>, Tensor<T>, …) inherits:
//        class Matrix : public Core<Matrix>
//
//    CRTP gives:
//      • Static dispatch to Derived methods (no vtable).
//      • Uniform device / dtype / storage access.
//      • Hooks for future autograd metadata, etc.
// ============================================================

template <typename Derived>
class Core {
public:
    // ---- construction ----------------------------------------

    Core() = default;

    Core(DType dtype, Device device)
        : dtype_(dtype), device_(device) {}

    // ---- type / device queries --------------------------------

    DType   dtype()   const { return dtype_;  }
    Device  device()  const { return device_; }

    bool is_cpu()     const { return device_.is_cpu();  }
    bool is_cuda()    const { return device_.is_cuda(); }
    bool is_mps()     const { return device_.is_mps();  }

    // ---- CRTP self() helper ----------------------------------
    //      Lets Core call Derived methods without virtuals.

    Derived&       self()       { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

    // ---- Storage management ----------------------------------
    //      Derived classes call init_storage() once they know
    //      the required byte count.

    void init_storage(std::size_t bytes) {
        storage_ = std::make_shared<Storage>(bytes, device_);
    }

    // Shared storage for zero-copy views / slices
    std::shared_ptr<Storage> storage() const { return storage_; }

    bool has_storage() const { return storage_ != nullptr; }

    // Typed raw pointer — convenience for Derived
    template <typename T>
    T* data_ptr() {
        if (!storage_) throw std::runtime_error("No storage allocated");
        return static_cast<T*>(storage_->data());
    }

    template <typename T>
    const T* data_ptr() const {
        if (!storage_) throw std::runtime_error("No storage allocated");
        return static_cast<const T*>(storage_->data());
    }

    // ---- Device transfer hook --------------------------------
    //      Derived should override to_device() for real copies;
    //      this base version is a no-op guard.

    Derived to(Device target) const {
        if (device_ == target) return self();
        throw std::runtime_error(
            "to(" + target.str() + ") not implemented for this type");
    }

    // ---- Metadata string (for printing / debugging) ----------

    std::string meta_str() const {
        return "dtype=" + dtype_str(dtype_) +
               " device=" + device_.str();
    }

protected:
    DType                    dtype_   = DType::Float32;
    Device                   device_  = Device::cpu();
    std::shared_ptr<Storage> storage_ = nullptr;

    // Future slots — add without breaking existing Derived classes:
    // bool       requires_grad_ = false;
    // GradFn     grad_fn_       = nullptr;
    // std::string name_         = "";

    static std::size_t compute_numel(const std::vector<std::size_t>& shape) {
        std::size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

} // namespace mlf