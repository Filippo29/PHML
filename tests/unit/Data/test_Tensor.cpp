#include "PHML/Data/Tensor.hpp"

#include <iostream>

using namespace PHML::Data;

int test_matrix_multiplication();
int test_matrix_addition();
int test_matrix_subtraction();

int main() {
    if (test_matrix_multiplication() != 0) return 1;
    if (test_matrix_addition() != 0) return 1;
    if (test_matrix_subtraction() != 0) return 1;

    return 0;
}

int test_matrix_multiplication() {
    Matrix A(2, 3, 1.0);
    Matrix B(3, 2, 2.0);

    Matrix C = A * B;

    Matrix expected(2, 2);
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
    Matrix A(2, 2, 1.0);
    Matrix B(2, 2, 2.0);

    Matrix C = A + B;

    Matrix expected(2, 2);
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
    Matrix A(2, 2, 5.0);
    Matrix B(2, 2, 3.0);

    Matrix C = A - B;

    Matrix expected(2, 2);
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