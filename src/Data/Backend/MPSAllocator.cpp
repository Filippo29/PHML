#ifdef PHML_WITH_MPS

#include <Metal/Metal.hpp>

#include "PHML/Data/Core.hpp"
#include "PHML/Data/Backend/MPSAllocator.hpp"
#include "PHML/Data/Backend/MetalContext.hpp"

#include <stdexcept>

namespace PHML::Data {

class MPSAllocator final : public Allocator {
public:
    MPSAllocator()  = default;
    ~MPSAllocator() override = default;

    MPSAllocator(const MPSAllocator&)            = delete;
    MPSAllocator& operator=(const MPSAllocator&) = delete;

    void* allocate(std::size_t bytes) override {
        MetalContext& ctx = MetalContext::instance();
        MTL::Buffer* buf = ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (!buf) throw std::bad_alloc();
        void* ptr = buf->contents();
        ctx.register_buffer(ptr, buf);
        return ptr;
    }

    void deallocate(void* ptr, std::size_t /*bytes*/) override {
        MetalContext& ctx = MetalContext::instance();
        MTL::Buffer* buf = ctx.buffer_for(ptr);
        ctx.unregister_buffer(ptr);
        buf->release();
    }

    Device device() const override { return Device::mps(); }
};

void register_mps_allocator(AllocatorRegistry& registry) {
    registry.register_allocator(Device::mps(), std::make_shared<MPSAllocator>());
}

} // namespace PHML::Data

#endif // PHML_WITH_MPS
