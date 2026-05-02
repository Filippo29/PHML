#include "PHML/Data/Core.hpp"
#include "PHML/Data/Matrix.hpp"
#include "PHML/Data/Tensor.hpp"

#include <cmath>
#include <iostream>

using namespace PHML::Data;

template <typename T>
static bool approx_eq(T a, T b, T tol = T{1e-9}) {
    return std::abs(a - b) <= tol * std::max(T{1}, std::max(std::abs(a), std::abs(b)));
}

#ifdef PHML_WITH_MPS

int test_mps_device_identity() {
    Device d = Device::mps();
    if (!d.is_mps())    { std::cerr << "mps device: is_mps() false\n";    return 1; }
    if (d.is_cpu())     { std::cerr << "mps device: is_cpu() true\n";     return 1; }
    if (!d.str().starts_with("mps")) { std::cerr << "mps device: str() != \"mps\"\n"; return 1; }

    Device d1 = Device::mps(1);
    if (d1.str() != "mps:1") { std::cerr << "mps:1 device: str() wrong\n"; return 1; }
    if (d1 == d)  { std::cerr << "mps:0 == mps:1 should be false\n"; return 1; }

    return 0;
}

int test_mps_tensor_allocation() {
    Tensor<float> t({2, 3}, Device::mps());
    if (!t.is_mps())    { std::cerr << "mps tensor: is_mps() false\n";    return 1; }
    if (t.numel() != 6) { std::cerr << "mps tensor: numel wrong\n";       return 1; }
    if (t.dtype() != DType::Float32) { std::cerr << "mps tensor: dtype wrong\n"; return 1; }

    for (auto v : t)
        if (v != 0.0f) { std::cerr << "mps tensor: default value not zero\n"; return 1; }

    t(0, 0) = 1.5f; t(1, 2) = 7.0f;
    if (t(0, 0) != 1.5f || t(1, 2) != 7.0f) {
        std::cerr << "mps tensor: write/read failed\n"; return 1;
    }

    return 0;
}

int test_mps_matrix_ops() {
    Matrix<double> m(2, 2, Device::mps());
    if (!m.is_mps()) { std::cerr << "mps matrix: is_mps() false\n"; return 1; }

    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    if (m(0, 0) != 1.0 || m(1, 1) != 4.0) {
        std::cerr << "mps matrix: element read/write failed\n"; return 1;
    }

    Matrix<double> n(2, 2, Device::mps());
    n(0, 0) = 1.0; n(0, 1) = 0.0;
    n(1, 0) = 0.0; n(1, 1) = 1.0;

    auto sum = m + n;
    if (!sum.is_mps()) { std::cerr << "mps matrix add: result not on mps\n"; return 1; }
    if (sum(0, 0) != 2.0 || sum(1, 1) != 5.0) {
        std::cerr << "mps matrix add: values wrong\n"; return 1;
    }

    return 0;
}

int test_mps_tensor_transfer_roundtrip() {
    Tensor<float> cpu_src({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    Tensor<float> on_mps = cpu_src.to(Device::mps());
    if (!on_mps.is_mps()) { std::cerr << "transfer to mps: is_mps() false\n"; return 1; }
    if (on_mps.storage().get() == cpu_src.storage().get()) {
        std::cerr << "transfer to mps: shares storage with source\n"; return 1;
    }
    for (std::size_t i = 0; i < cpu_src.numel(); ++i)
        if (on_mps.begin()[i] != cpu_src.begin()[i]) {
            std::cerr << "transfer to mps: value mismatch at " << i << "\n"; return 1;
        }

    Tensor<float> back = on_mps.to(Device::cpu());
    if (!back.is_cpu()) { std::cerr << "transfer back to cpu: is_cpu() false\n"; return 1; }
    for (std::size_t i = 0; i < cpu_src.numel(); ++i)
        if (back.begin()[i] != cpu_src.begin()[i]) {
            std::cerr << "transfer back to cpu: value mismatch at " << i << "\n"; return 1;
        }

    return 0;
}

int test_mps_tensor_arithmetic() {
    Tensor<double> a_cpu({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor<double> b_cpu({2, 2}, {5.0, 6.0, 7.0, 8.0});
    Tensor<double> expected = a_cpu + b_cpu;

    Tensor<double> a_mps = a_cpu.to(Device::mps());
    Tensor<double> b_mps = b_cpu.to(Device::mps());
    Tensor<double> result_mps = a_mps + b_mps;

    if (!result_mps.is_mps()) { std::cerr << "mps arith: result not on mps\n"; return 1; }
    for (std::size_t i = 0; i < expected.numel(); ++i)
        if (result_mps.begin()[i] != expected.begin()[i]) {
            std::cerr << "mps arith: value mismatch at " << i << "\n"; return 1;
        }

    return 0;
}

// ---- GPU compute tests --------------------------------------------------

int test_mps_gpu_matmul_f32() {
    // Build inputs on CPU with known values.
    Matrix<float> A_cpu(2, 3);
    A_cpu(0,0)=1; A_cpu(0,1)=2; A_cpu(0,2)=3;
    A_cpu(1,0)=4; A_cpu(1,1)=5; A_cpu(1,2)=6;

    Matrix<float> B_cpu(3, 2);
    B_cpu(0,0)=7;  B_cpu(0,1)=8;
    B_cpu(1,0)=9;  B_cpu(1,1)=10;
    B_cpu(2,0)=11; B_cpu(2,1)=12;

    Matrix<float> ref = A_cpu * B_cpu;  // CPU reference

    Matrix<float> A_mps = A_cpu.to(Device::mps());
    Matrix<float> B_mps = B_cpu.to(Device::mps());
    Matrix<float> C_mps = A_mps * B_mps;

    if (!C_mps.is_mps()) { std::cerr << "gpu matmul: result not on mps\n"; return 1; }
    if (C_mps.rows() != 2 || C_mps.cols() != 2) { std::cerr << "gpu matmul: shape wrong\n"; return 1; }

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            if (!approx_eq(C_mps(i, j), ref(i, j))) {
                std::cerr << "gpu matmul: value mismatch at (" << i << "," << j
                          << "): got " << C_mps(i, j) << " expected " << ref(i, j) << "\n";
                return 1;
            }
    return 0;
}

int test_mps_gpu_elementwise_f32() {
    Tensor<float> a_cpu({3, 4}, {1,2,3,4,5,6,7,8,9,10,11,12});
    Tensor<float> b_cpu({3, 4}, {12,11,10,9,8,7,6,5,4,3,2,1});

    Tensor<float> a_mps = a_cpu.to(Device::mps());
    Tensor<float> b_mps = b_cpu.to(Device::mps());

    // addition
    auto ref_add  = a_cpu + b_cpu;
    auto mps_add  = a_mps + b_mps;
    if (!mps_add.is_mps()) { std::cerr << "gpu elementwise: add result not on mps\n"; return 1; }
    for (std::size_t i = 0; i < ref_add.numel(); ++i)
        if (!approx_eq(mps_add.begin()[i], ref_add.begin()[i])) {
            std::cerr << "gpu elementwise: add mismatch at " << i << "\n"; return 1;
        }

    // subtraction
    auto ref_sub  = a_cpu - b_cpu;
    auto mps_sub  = a_mps - b_mps;
    for (std::size_t i = 0; i < ref_sub.numel(); ++i)
        if (!approx_eq(mps_sub.begin()[i], ref_sub.begin()[i])) {
            std::cerr << "gpu elementwise: sub mismatch at " << i << "\n"; return 1;
        }

    // Hadamard product
    auto ref_mul  = a_cpu * b_cpu;
    auto mps_mul  = a_mps * b_mps;
    for (std::size_t i = 0; i < ref_mul.numel(); ++i)
        if (!approx_eq(mps_mul.begin()[i], ref_mul.begin()[i])) {
            std::cerr << "gpu elementwise: hadamard mismatch at " << i << "\n"; return 1;
        }

    // scalar multiply
    auto ref_scale = a_cpu * 2.5f;
    auto mps_scale = a_mps * 2.5f;
    for (std::size_t i = 0; i < ref_scale.numel(); ++i)
        if (!approx_eq(mps_scale.begin()[i], ref_scale.begin()[i])) {
            std::cerr << "gpu elementwise: scale mismatch at " << i << "\n"; return 1;
        }

    return 0;
}

int test_mps_double_fallback() {
    // double matmul on MPS triggers the CPU fallback path (no GPU kernel for f64).
    // The result must still be numerically correct.
    Matrix<double> A(3, 3, 1.0, Device::mps());
    Matrix<double> B(3, 3, 2.0, Device::mps());
    Matrix<double> C = A * B;   // hits warn_fallback_once, runs CPU loop

    if (!C.is_mps()) { std::cerr << "double fallback: result not on mps\n"; return 1; }
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (!approx_eq(C(i, j), 6.0)) {
                std::cerr << "double fallback: value wrong at (" << i << "," << j << ")\n";
                return 1;
            }
    return 0;
}

#endif // PHML_WITH_MPS

int main() {
#ifdef PHML_WITH_MPS
    if (test_mps_device_identity()           != 0) return 1;
    if (test_mps_tensor_allocation()         != 0) return 1;
    if (test_mps_matrix_ops()                != 0) return 1;
    if (test_mps_tensor_transfer_roundtrip() != 0) return 1;
    if (test_mps_tensor_arithmetic()         != 0) return 1;
    if (test_mps_gpu_matmul_f32()            != 0) return 1;
    if (test_mps_gpu_elementwise_f32()       != 0) return 1;
    if (test_mps_double_fallback()           != 0) return 1;
#endif
    return 0;
}
