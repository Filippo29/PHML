#ifdef PHML_WITH_MPS

#include <Metal/Metal.hpp>

#include "PHML/Data/Backend/MPSOps.hpp"
#include "PHML/Data/Backend/MetalContext.hpp"

#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace PHML::Data::mps {

// ---- shared dispatch helpers --------------------------------------------

static void commit_and_wait(MTL::CommandBuffer* cmd) {
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
}

template <typename Fn>
static void encode_and_run(Fn fn) {
    MTL::CommandBuffer* cmd = MetalContext::instance().queue()->commandBuffer();
    if (!cmd) throw std::runtime_error("MPSOps: failed to create command buffer");
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    if (!enc) {
        cmd->release();
        throw std::runtime_error("MPSOps: failed to create compute encoder");
    }
    fn(enc);
    enc->endEncoding();
    commit_and_wait(cmd);
}

// ---- matmul -------------------------------------------------------------

void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t K, std::size_t N) {
    MetalContext& ctx = MetalContext::instance();
    MTL::Buffer* bufA = ctx.buffer_for(A);
    MTL::Buffer* bufB = ctx.buffer_for(B);
    MTL::Buffer* bufC = ctx.buffer_for(C);
    MTL::ComputePipelineState* pso = ctx.pipeline("matmul_f32");

    const uint32_t dims[3] = {(uint32_t)M, (uint32_t)K, (uint32_t)N};

    encode_and_run([&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA, 0, 0);
        enc->setBuffer(bufB, 0, 1);
        enc->setBuffer(bufC, 0, 2);
        enc->setBytes(dims, sizeof(dims), 3);
        // One thread per output element; Apple Silicon supports non-uniform grids.
        constexpr NS::UInteger TS = 8;
        enc->dispatchThreads(MTL::Size(N, M, 1), MTL::Size(TS, TS, 1));
    });
}

// ---- element-wise -------------------------------------------------------

void add_f32(const float* a, const float* b, float* out, std::size_t n) {
    MetalContext& ctx = MetalContext::instance();
    MTL::Buffer* bufA   = ctx.buffer_for(a);
    MTL::Buffer* bufB   = ctx.buffer_for(b);
    MTL::Buffer* bufOut = ctx.buffer_for(out);
    MTL::ComputePipelineState* pso = ctx.pipeline("add_f32");
    const uint32_t count = (uint32_t)n;

    encode_and_run([&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA,   0, 0);
        enc->setBuffer(bufB,   0, 1);
        enc->setBuffer(bufOut, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    });
}

void sub_f32(const float* a, const float* b, float* out, std::size_t n) {
    MetalContext& ctx = MetalContext::instance();
    MTL::Buffer* bufA   = ctx.buffer_for(a);
    MTL::Buffer* bufB   = ctx.buffer_for(b);
    MTL::Buffer* bufOut = ctx.buffer_for(out);
    MTL::ComputePipelineState* pso = ctx.pipeline("sub_f32");
    const uint32_t count = (uint32_t)n;

    encode_and_run([&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA,   0, 0);
        enc->setBuffer(bufB,   0, 1);
        enc->setBuffer(bufOut, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    });
}

void mul_f32(const float* a, const float* b, float* out, std::size_t n) {
    MetalContext& ctx = MetalContext::instance();
    MTL::Buffer* bufA   = ctx.buffer_for(a);
    MTL::Buffer* bufB   = ctx.buffer_for(b);
    MTL::Buffer* bufOut = ctx.buffer_for(out);
    MTL::ComputePipelineState* pso = ctx.pipeline("mul_f32");
    const uint32_t count = (uint32_t)n;

    encode_and_run([&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA,   0, 0);
        enc->setBuffer(bufB,   0, 1);
        enc->setBuffer(bufOut, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    });
}

void scale_f32(const float* a, float scalar, float* out, std::size_t n) {
    MetalContext& ctx = MetalContext::instance();
    MTL::Buffer* bufA   = ctx.buffer_for(a);
    MTL::Buffer* bufOut = ctx.buffer_for(out);
    MTL::ComputePipelineState* pso = ctx.pipeline("scale_f32");
    const uint32_t count = (uint32_t)n;

    encode_and_run([&](MTL::ComputeCommandEncoder* enc) {
        enc->setComputePipelineState(pso);
        enc->setBuffer(bufA,   0, 0);
        enc->setBytes(&scalar, sizeof(scalar), 1);
        enc->setBuffer(bufOut, 0, 2);
        enc->setBytes(&count, sizeof(count), 3);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    });
}

// ---- fallback notice ----------------------------------------------------

void warn_fallback_once(const char* op, const std::string& dtype) {
    static std::mutex mu;
    static std::unordered_set<std::string> seen;
    const std::string key = std::string(op) + ":" + dtype;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(key).second)
        std::cerr << "[PHML::MPS] '" << op << "' has no GPU implementation for dtype '"
                  << dtype << "'; running on CPU via shared-memory pointer "
                  << "(this notice is shown once).\n";
}

} // namespace PHML::Data::mps

#endif // PHML_WITH_MPS
