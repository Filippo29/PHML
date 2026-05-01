#include "PHML/Data/Core.hpp"
#include "PHML/Data/Tensor.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace PHML::Data;

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

    auto tp = t.transpose(0, 1);
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
    auto tp = t.transpose(0, 1);
    auto c  = tp.contiguous();
    if (!c.is_contiguous()) { std::cerr << "contiguous() result not contiguous\n"; return 1; }
    if (c.shape() != tp.shape()) { std::cerr << "contiguous() shape wrong\n"; return 1; }
    if (c.storage().get() == t.storage().get()) { std::cerr << "contiguous() did not copy\n"; return 1; }

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            if (c(i, j) != tp(i, j)) { std::cerr << "contiguous() value mismatch\n"; return 1; }

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

int main() {
    if (test_tensor_construction()      != 0) return 1;
    if (test_tensor_element_access()    != 0) return 1;
    if (test_tensor_arithmetic()        != 0) return 1;
    if (test_tensor_reshape()           != 0) return 1;
    if (test_tensor_permute_transpose() != 0) return 1;
    if (test_tensor_contiguous()        != 0) return 1;
    if (test_tensor_factories()         != 0) return 1;
    return 0;
}
