#include <iostream>
#include <chrono>
#include <cstddef>

#include "PHML/PHML.hpp"
#include "PHML/Data/Matrix.hpp"
#include "PHML/Data/Tensor.hpp"

using namespace PHML::Data;
using Clock = std::chrono::high_resolution_clock;

template <typename T>
static long long bench_matmul(const Matrix<T>& a, const Matrix<T>& b, int reps) {
    long long total = 0;
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        volatile auto r = a * b;
        auto t1 = Clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    return total / reps;
}

int main() {
    std::cout << "PHML version: " << PHML::core::version_string() << "\n\n";

    constexpr std::size_t N    = 256;
    constexpr int         REPS = 5;

    // ---- double benchmark (CPU only — no GPU kernel for f64) ------------
    std::cout << "=== Matrix<double> " << N << "x" << N << " ===\n";

    Matrix<double> cpu_d = Matrix<double>::random(N, N, 0.0, 1.0, Device::cpu());
    Matrix<double> cpu_d2 = Matrix<double>::random(N, N, 0.0, 1.0, Device::cpu());
    std::cout << "  CPU : " << bench_matmul(cpu_d, cpu_d2, REPS) << " µs\n";

#ifdef PHML_WITH_MPS
    Matrix<double> mps_d  = cpu_d.to(Device::mps());
    Matrix<double> mps_d2 = cpu_d2.to(Device::mps());
    std::cout << "  MPS : " << bench_matmul(mps_d, mps_d2, REPS)
              << " µs  (fallback — no GPU kernel for float64)\n";
#endif

    // ---- float benchmark (GPU-accelerated on MPS) -----------------------
    std::cout << "\n=== Matrix<float> " << N << "x" << N << " ===\n";

    Matrix<float> cpu_f  = Matrix<float>::random(N, N, 0.f, 1.f, Device::cpu());
    Matrix<float> cpu_f2 = Matrix<float>::random(N, N, 0.f, 1.f, Device::cpu());
    long long cpu_f_us = bench_matmul(cpu_f, cpu_f2, REPS);
    std::cout << "  CPU : " << cpu_f_us << " µs\n";

#ifdef PHML_WITH_MPS
    Matrix<float> mps_f  = cpu_f.to(Device::mps());
    Matrix<float> mps_f2 = cpu_f2.to(Device::mps());
    long long mps_f_us = bench_matmul(mps_f, mps_f2, REPS);
    std::cout << "  MPS : " << mps_f_us << " µs  (GPU dispatch)\n";
    std::cout << "  Speedup: "
              << static_cast<double>(cpu_f_us) / static_cast<double>(mps_f_us) << "x\n";
#endif

    return 0;
}
