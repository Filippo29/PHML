#include "PHML/Data/Core.hpp"
#include "PHML/Data/Matrix.hpp"
#include "PHML/Data/Tensor.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <iomanip>

using namespace PHML::Data;

// Combined absolute + relative tolerance comparison for floating-point values.
// Relative handles large magnitudes; absolute handles values near zero.
template <typename T>
static bool approx_equal(T a, T b,
                         T abs_tol = T{1e-9},
                         T rel_tol = T{1e-9}) {
    T diff = std::abs(a - b);
    return diff <= std::max(abs_tol,
                            rel_tol * std::max(std::abs(a), std::abs(b)));
}

int test_matrix_multiplication();
int test_matrix_addition();
int test_matrix_subtraction();
int test_matrix_determinant();

int test_tensor_construction();
int test_tensor_element_access();
int test_tensor_arithmetic();
int test_tensor_reshape();
int test_tensor_permute_transpose();
int test_tensor_contiguous();
int test_tensor_factories();

int main() {
    if (test_matrix_multiplication() != 0) return 1;
    if (test_matrix_addition() != 0) return 1;
    if (test_matrix_subtraction() != 0) return 1;
    if (test_matrix_determinant() != 0) return 1;

    if (test_tensor_construction()      != 0) return 1;
    if (test_tensor_element_access()    != 0) return 1;
    if (test_tensor_arithmetic()        != 0) return 1;
    if (test_tensor_reshape()           != 0) return 1;
    if (test_tensor_permute_transpose() != 0) return 1;
    if (test_tensor_contiguous()        != 0) return 1;
    if (test_tensor_factories()         != 0) return 1;

    return 0;
}

int test_matrix_multiplication() {
    Matrix<double> A(2, 3, 1.0);
    Matrix<double> B(3, 2, 2.0);

    Matrix<double> C = A * B;

    Matrix<double> expected(2, 2);
    expected(0, 0) = 6.0;
    expected(0, 1) = 6.0;
    expected(1, 0) = 6.0;
    expected(1, 1) = 6.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (C(i, j) != expected(i, j)) {
                std::cerr << "Test failed at (" << i << ", " << j << "): "
                          << "expected " << expected(i, j) << ", got " << C(i, j) << '\n';
                return 1;
            }
        }
    }

    return 0;
}

int test_matrix_addition() {
    Matrix<double> A(2, 2, 1.0);
    Matrix<double> B(2, 2, 2.0);

    Matrix<double> C = A + B;

    Matrix<double> expected(2, 2);
    expected(0, 0) = 3.0;
    expected(0, 1) = 3.0;
    expected(1, 0) = 3.0;
    expected(1, 1) = 3.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (C(i, j) != expected(i, j)) {
                std::cerr << "Test failed at (" << i << ", " << j << "): "
                          << "expected " << expected(i, j) << ", got " << C(i, j) << '\n';
                return 1;
            }
        }
    }

    return 0;
}

int test_matrix_subtraction() {
    Matrix<double> A(2, 2, 5.0);
    Matrix<double> B(2, 2, 3.0);

    Matrix<double> C = A - B;

    Matrix<double> expected(2, 2);
    expected(0, 0) = 2.0;
    expected(0, 1) = 2.0;
    expected(1, 0) = 2.0;
    expected(1, 1) = 2.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (C(i, j) != expected(i, j)) {
                std::cerr << "Test failed at (" << i << ", " << j << "): "
                          << "expected " << expected(i, j) << ", got " << C(i, j) << '\n';
                return 1;
            }
        }
    }

    return 0;
}

int test_matrix_determinant() {
    // test 1 negative determinant
    Matrix<double> A(2, 2);
    A(0, 0) = 4.0; A(0, 1) = 3.0;
    A(1, 0) = 6.0; A(1, 1) = 3.0;

    double detA = A.determinant();
    double expectedDetA = -6.0;

    if (!approx_equal(detA, expectedDetA)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for determinant: expected " << expectedDetA << ", got " << detA << '\n';
        return 1;
    }

    // test 2 positive determinant
    Matrix<double> B(2, 2);
    B(0, 0) = 1.0; B(0, 1) = 2.5;
    B(1, 0) = -3.0; B(1, 1) = 4.0;

    double detB = B.determinant();
    double expectedDetB = 11.5;

    if (!approx_equal(detB, expectedDetB)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for determinant: expected " << expectedDetB << ", got " << detB << '\n';
        return 1;
    }

    // test 3x3

    Matrix<double> C(3, 3);
    C(0, 0) = 6;  C(0, 1) = 1;  C(0, 2) = 1;
    C(1, 0) = 4;  C(1, 1) = -2; C(1, 2) = 5;
    C(2, 0) = 2;  C(2, 1) = 8;  C(2, 2) = 7;

    double detC = C.determinant();
    double expectedDetC = -306.0;

    if (!approx_equal(detC, expectedDetC)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for 3x3: expected " << expectedDetC << ", got " << detC << '\n';
        return 1;
    }

    // test 3x3 with zero determinant

    Matrix<double> D(3, 3);
    D(0, 0) = 1; D(0, 1) = 2; D(0, 2) = 3;
    D(1, 0) = 2; D(1, 1) = 4; D(1, 2) = 6; // row = 2 * row 0
    D(2, 0) = 7; D(2, 1) = 8; D(2, 2) = 9;

    double detD = D.determinant();
    double expectedDetD = 0.0;

    if (!approx_equal(detD, expectedDetD)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for singular matrix: expected 0, got " << detD << '\n';
        return 1;
    }

    // test 4x4

    Matrix<double> E(4, 4);
    E(0,0)=1; E(0,1)=2; E(0,2)=3; E(0,3)=4;
    E(1,0)=5; E(1,1)=6; E(1,2)=7; E(1,3)=8;
    E(2,0)=2; E(2,1)=6; E(2,2)=4; E(2,3)=8;
    E(3,0)=3; E(3,1)=1; E(3,2)=1; E(3,3)=2;

    double detE = E.determinant();
    double expectedDetE = 72.0;
    if (!approx_equal(detE, expectedDetE)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for 4x4: expected " << expectedDetE << ", got " << detE << '\n';
        return 1;
    }

    // test identity matrix, should always be 1

    Matrix<double> I(4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            I(i, j) = (i == j) ? 1.0 : 0.0;

    double detI = I.determinant();
    if (!approx_equal(detI, 1.0)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for identity matrix: expected 1, got " << detI << '\n';
        return 1;
    }

    // test upper triangular matrix (det = product of diagonal)

    Matrix<double> T(3, 3);
    T(0,0)=2; T(0,1)=3; T(0,2)=1;
    T(1,0)=0; T(1,1)=5; T(1,2)=4;
    T(2,0)=0; T(2,1)=0; T(2,2)=7;

    double detT = T.determinant();
    double expectedDetT = 2 * 5 * 7; // 70

    if (!approx_equal(detT, expectedDetT)) {
        std::cerr << std::setprecision(17)
                  << "Test failed for triangular matrix: expected " << expectedDetT << ", got " << detT << '\n';
        return 1;
    }

    return 0;
}

// ============================================================
// Tensor tests
// ============================================================

int test_tensor_construction() {
    Tensor<float> t({2, 3, 4});
    if (t.ndim() != 3) { std::cerr << "ndim != 3\n"; return 1; }
    if (t.shape() != std::vector<std::size_t>{2, 3, 4}) { std::cerr << "shape mismatch\n"; return 1; }
    if (t.strides() != std::vector<std::size_t>{12, 4, 1}) { std::cerr << "strides mismatch\n"; return 1; }
    if (t.numel() != 24) { std::cerr << "numel != 24\n"; return 1; }
    if (!t.is_contiguous()) { std::cerr << "freshly built tensor not contiguous\n"; return 1; }
    if (t.dtype() != DType::Float32) { std::cerr << "dtype mismatch\n"; return 1; }
    for (auto v : t) {
        if (v != 0.0f) { std::cerr << "default-constructed value != 0\n"; return 1; }
    }

    Tensor<double> filled({2, 2}, 3.14);
    for (auto v : filled) {
        if (v != 3.14) { std::cerr << "fill value mismatch\n"; return 1; }
    }

    Tensor<int32_t> lit({2, 2}, {1, 2, 3, 4});
    if (lit(0, 0) != 1 || lit(0, 1) != 2 || lit(1, 0) != 3 || lit(1, 1) != 4) {
        std::cerr << "init-list construction failed\n"; return 1;
    }
    return 0;
}

int test_tensor_element_access() {
    Tensor<int32_t> t({2, 3, 4});
    int32_t v = 0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                t(i, j, k) = v++;

    if (t(1, 2, 3) != 23) { std::cerr << "operator() read failed\n"; return 1; }
    if (t.at({0, 1, 2}) != 6) { std::cerr << "at() read failed\n"; return 1; }

    bool threw = false;
    try { (void)t(2, 0, 0); } catch (const std::out_of_range&) { threw = true; }
    if (!threw) { std::cerr << "out-of-range did not throw\n"; return 1; }

    threw = false;
    try { (void)t.at({0, 0}); } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) { std::cerr << "wrong-rank at() did not throw\n"; return 1; }

    return 0;
}

int test_tensor_arithmetic() {
    Tensor<double> a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor<double> b({2, 2}, {5.0, 6.0, 7.0, 8.0});

    auto sum = a + b;
    if (sum(0, 0) != 6.0 || sum(1, 1) != 12.0) { std::cerr << "add failed\n"; return 1; }

    auto diff = b - a;
    if (diff(0, 0) != 4.0 || diff(1, 1) != 4.0) { std::cerr << "sub failed\n"; return 1; }

    auto had = a * b;
    if (had(0, 0) != 5.0 || had(1, 1) != 32.0) { std::cerr << "hadamard failed\n"; return 1; }

    auto scaled = a * 2.0;
    if (scaled(0, 0) != 2.0 || scaled(1, 1) != 8.0) { std::cerr << "scalar * failed\n"; return 1; }

    auto scaled2 = 3.0 * a;
    if (scaled2(0, 0) != 3.0 || scaled2(1, 1) != 12.0) { std::cerr << "scalar * (lhs) failed\n"; return 1; }

    Tensor<double> wrong({3, 2});
    bool threw = false;
    try { (void)(a + wrong); } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) { std::cerr << "shape-mismatch add did not throw\n"; return 1; }

    return 0;
}

int test_tensor_reshape() {
    Tensor<int32_t> t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto r = t.reshape({3, 2});
    if (r.shape() != std::vector<std::size_t>{3, 2}) { std::cerr << "reshape shape wrong\n"; return 1; }
    if (r(0, 0) != 1 || r(2, 1) != 6) { std::cerr << "reshape values wrong\n"; return 1; }
    if (r.storage().get() != t.storage().get()) { std::cerr << "reshape not zero-copy\n"; return 1; }

    t(0, 0) = 99;
    if (r(0, 0) != 99) { std::cerr << "reshape does not share storage\n"; return 1; }

    bool threw = false;
    try { (void)t.reshape({5, 2}); } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) { std::cerr << "reshape numel-mismatch did not throw\n"; return 1; }

    auto tp = t.transpose(0, 1); // non-contiguous
    threw = false;
    try { (void)tp.reshape({6}); } catch (const std::runtime_error&) { threw = true; }
    if (!threw) { std::cerr << "reshape on non-contiguous did not throw\n"; return 1; }

    return 0;
}

int test_tensor_permute_transpose() {
    Tensor<int32_t> t({2, 3, 4});
    int32_t v = 0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                t(i, j, k) = v++;

    auto p = t.permute({2, 0, 1});
    if (p.shape() != std::vector<std::size_t>{4, 2, 3}) { std::cerr << "permute shape wrong\n"; return 1; }
    if (p.is_contiguous()) { std::cerr << "permuted view unexpectedly contiguous\n"; return 1; }
    if (p.storage().get() != t.storage().get()) { std::cerr << "permute not zero-copy\n"; return 1; }

    // p(a, b, c) should equal t(b, c, a)
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t c = 0; c < 3; ++c)
                if (p(a, b, c) != t(b, c, a)) {
                    std::cerr << "permute value mapping wrong\n"; return 1;
                }

    auto tr = t.transpose(0, 2);
    if (tr.shape() != std::vector<std::size_t>{4, 3, 2}) { std::cerr << "transpose shape wrong\n"; return 1; }
    if (tr(1, 2, 0) != t(0, 2, 1)) { std::cerr << "transpose value wrong\n"; return 1; }

    return 0;
}

int test_tensor_contiguous() {
    Tensor<double> t({2, 3}, {1, 2, 3, 4, 5, 6});
    auto tp = t.transpose(0, 1); // shape {3, 2}, non-contiguous
    auto c  = tp.contiguous();
    if (!c.is_contiguous()) { std::cerr << "contiguous() result not contiguous\n"; return 1; }
    if (c.shape() != tp.shape()) { std::cerr << "contiguous() shape wrong\n"; return 1; }
    if (c.storage().get() == t.storage().get()) { std::cerr << "contiguous() did not copy\n"; return 1; }

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            if (c(i, j) != tp(i, j)) { std::cerr << "contiguous() value mismatch\n"; return 1; }

    // Mutating the source must not affect the materialised copy
    t(0, 0) = 999;
    if (c(0, 0) == 999) { std::cerr << "contiguous() still aliases source\n"; return 1; }

    return 0;
}

int test_tensor_factories() {
    auto z = Tensor<float>::zeros({2, 3});
    for (auto v : z) if (v != 0.0f) { std::cerr << "zeros failed\n"; return 1; }

    auto o = Tensor<float>::ones({2, 3});
    for (auto v : o) if (v != 1.0f) { std::cerr << "ones failed\n"; return 1; }

    auto f = Tensor<int32_t>::full({2, 2}, 7);
    for (auto v : f) if (v != 7) { std::cerr << "full failed\n"; return 1; }

    auto r = Tensor<double>::random({3, 3}, 0.0, 1.0);
    if (r.shape() != std::vector<std::size_t>{3, 3}) { std::cerr << "random shape wrong\n"; return 1; }
    for (auto v : r) if (v < 0.0 || v > 1.0) { std::cerr << "random out of range\n"; return 1; }

    AnyTensor at = TensorFactory::random({2, 2}, DType::Float64, -1.0, 1.0);
    auto& td = std::get<Tensor<double>>(at);
    if (td.shape() != std::vector<std::size_t>{2, 2}) { std::cerr << "TensorFactory::random shape wrong\n"; return 1; }

    AnyTensor az = TensorFactory::zeros({2, 2}, DType::Int32);
    auto& ti = std::get<Tensor<int32_t>>(az);
    for (auto v : ti) if (v != 0) { std::cerr << "TensorFactory::zeros failed\n"; return 1; }

    return 0;
}