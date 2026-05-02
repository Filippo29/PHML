#ifdef PHML_WITH_MPS

// metal-cpp private implementations live exactly once in this TU.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "PHML/Data/Backend/MetalContext.hpp"

#include <stdexcept>
#include <string>

namespace PHML::Data {

// MSL source for element-wise compute kernels (all four in one library).
static constexpr const char* kElementwiseMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device       float* out [[buffer(2)]],
                    constant     uint&  n   [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
    if (tid < n) out[tid] = a[tid] + b[tid];
}

kernel void sub_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device       float* out [[buffer(2)]],
                    constant     uint&  n   [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
    if (tid < n) out[tid] = a[tid] - b[tid];
}

kernel void mul_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device       float* out [[buffer(2)]],
                    constant     uint&  n   [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
    if (tid < n) out[tid] = a[tid] * b[tid];
}

kernel void scale_f32(device const float*  a [[buffer(0)]],
                      constant float&      s [[buffer(1)]],
                      device       float* out [[buffer(2)]],
                      constant     uint&   n  [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid < n) out[tid] = a[tid] * s;
}

// Naive GEMM: C[MxN] = A[MxK] * B[KxN], row-major.
// One thread per output element; Apple Silicon supports non-uniform grids.
kernel void matmul_f32(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device       float* C [[buffer(2)]],
                       constant     uint3& d [[buffer(3)]],  // {M, K, N}
                       uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= d.x || col >= d.z) return;
    float acc = 0.0f;
    for (uint k = 0; k < d.y; ++k)
        acc += A[row * d.y + k] * B[k * d.z + col];
    C[row * d.z + col] = acc;
}
)MSL";

// ---- singleton ----------------------------------------------------------

MetalContext::MetalContext() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_)
        throw std::runtime_error("MetalContext: no Metal device found");

    queue_ = device_->newCommandQueue();
    if (!queue_)
        throw std::runtime_error("MetalContext: failed to create command queue");
}

MetalContext::~MetalContext() {
    for (auto& [name, pso] : pipelines_)
        if (pso) pso->release();
    if (library_) library_->release();
    if (queue_)   queue_->release();
    if (device_)  device_->release();
}

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

// ---- accessors ----------------------------------------------------------

MTL::Device*       MetalContext::device() { return device_; }
MTL::CommandQueue* MetalContext::queue()  { return queue_;  }

MTL::Library* MetalContext::library_unlocked_() {
    if (library_) return library_;

    NS::Error*  err  = nullptr;
    NS::String* src  = NS::String::string(kElementwiseMSL, NS::UTF8StringEncoding);
    auto*       opts = MTL::CompileOptions::alloc()->init();
    library_ = device_->newLibrary(src, opts, &err);
    opts->release();

    if (!library_) {
        std::string msg = "MetalContext: failed to compile element-wise shaders";
        if (err) msg += ": " + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }
    return library_;
}

MTL::Library* MetalContext::elementwise_library() {
    std::lock_guard<std::mutex> lock(mutex_);
    return library_unlocked_();
}

MTL::ComputePipelineState* MetalContext::pipeline(const char* fn_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pipelines_.find(fn_name);
    if (it != pipelines_.end()) return it->second;

    MTL::Library* lib = library_unlocked_();

    NS::String*  name = NS::String::string(fn_name, NS::UTF8StringEncoding);
    MTL::Function* fn = lib->newFunction(name);
    if (!fn)
        throw std::runtime_error(
            std::string("MetalContext: no Metal function '") + fn_name + "'");

    NS::Error* err = nullptr;
    auto* pso = device_->newComputePipelineState(fn, &err);
    fn->release();

    if (!pso) {
        std::string msg =
            std::string("MetalContext: failed to create pipeline for '") + fn_name + "'";
        if (err) msg += ": " + std::string(err->localizedDescription()->utf8String());
        throw std::runtime_error(msg);
    }

    pipelines_[fn_name] = pso;
    return pso;
}

// ---- buffer registry ----------------------------------------------------

void MetalContext::register_buffer(void* ptr, MTL::Buffer* buf) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[ptr] = buf;
}

void MetalContext::unregister_buffer(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.erase(ptr);
}

MTL::Buffer* MetalContext::buffer_for(const void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end())
        throw std::runtime_error(
            "MetalContext: no MTLBuffer registered for pointer");
    return it->second;
}

} // namespace PHML::Data

#endif // PHML_WITH_MPS
