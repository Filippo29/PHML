#pragma once

#ifdef PHML_WITH_MPS

#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace PHML::Data {

// Process-wide Metal state: device, command queue, element-wise shader library,
// pipeline cache, and the void* -> MTLBuffer registry shared with MPSAllocator.
//
// Only included from .cpp files under src/Data/Backend/ — never from public headers.
class MetalContext {
public:
    static MetalContext& instance();

    MTL::Device*       device();
    MTL::CommandQueue* queue();
    MTL::Library*      elementwise_library();
    MTL::ComputePipelineState* pipeline(const char* fn_name);

    void         register_buffer(void* ptr, MTL::Buffer* buf);
    void         unregister_buffer(void* ptr);
    MTL::Buffer* buffer_for(const void* ptr);

    MetalContext(const MetalContext&)            = delete;
    MetalContext& operator=(const MetalContext&) = delete;

private:
    MetalContext();
    ~MetalContext();

    MTL::Library* library_unlocked_();  // caller must hold mutex_

    MTL::Device*       device_  = nullptr;
    MTL::CommandQueue* queue_   = nullptr;
    MTL::Library*      library_ = nullptr;

    std::mutex mutex_;
    std::unordered_map<const void*, MTL::Buffer*>             buffers_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
};

} // namespace PHML::Data

#endif // PHML_WITH_MPS
