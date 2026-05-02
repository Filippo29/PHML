#pragma once
#ifdef PHML_WITH_MPS
#include <cstddef>
#include <string>
namespace PHML::Data::mps {

void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t K, std::size_t N);

void add_f32  (const float* a, const float* b, float* out, std::size_t n);
void sub_f32  (const float* a, const float* b, float* out, std::size_t n);
void mul_f32  (const float* a, const float* b, float* out, std::size_t n);  // Hadamard
void scale_f32(const float* a, float scalar,   float* out, std::size_t n);

// Prints exactly once per (op, dtype) pair to std::cerr; thread-safe.
void warn_fallback_once(const char* op, const std::string& dtype);

} // namespace PHML::Data::mps
#endif