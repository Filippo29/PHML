#include "PHML/Data/Core.hpp"
#include "PHML/Data/Tensor.hpp"

#include <iostream>

using namespace PHML::Data;

int test_matrix_multiplication();
int test_matrix_addition();
int test_matrix_subtraction();
int test_matrix_determinant();

int main() {
    if (test_matrix_multiplication() != 0) return 1;
    // if (test_matrix_addition() != 0) return 1;
    // if (test_matrix_subtraction() != 0) return 1;
    // if (test_matrix_determinant() != 0) return 1;

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

    if (detA != expectedDetA) {
        std::cerr << "Test failed for determinant: expected " << expectedDetA << ", got " << detA << '\n';
        return 1;
    }

    // test 2 positive determinant
    Matrix<double> B(2, 2);
    B(0, 0) = 1.0; B(0, 1) = 2.5;
    B(1, 0) = -3.0; B(1, 1) = 4.0;

    double detB = B.determinant();
    double expectedDetB = 11.5;

    if (detB != expectedDetB) {
        std::cerr << "Test failed for determinant: expected " << expectedDetB << ", got " << detB << '\n';
        return 1;
    }

    // test 3x3

    Matrix<double> C(3, 3);
    C(0, 0) = 6;  C(0, 1) = 1;  C(0, 2) = 1;
    C(1, 0) = 4;  C(1, 1) = -2; C(1, 2) = 5;
    C(2, 0) = 2;  C(2, 1) = 8;  C(2, 2) = 7;

    double detC = C.determinant();
    double expectedDetC = -306.0;

    if (detC != expectedDetC) {
        std::cerr << "Test failed for 3x3: expected " << expectedDetC << ", got " << detC << '\n';
        return 1;
    }

    // test 3x3 with zero determinant

    Matrix<double> D(3, 3);
    D(0, 0) = 1; D(0, 1) = 2; D(0, 2) = 3;
    D(1, 0) = 2; D(1, 1) = 4; D(1, 2) = 6; // row = 2 * row 0
    D(2, 0) = 7; D(2, 1) = 8; D(2, 2) = 9;

    double detD = D.determinant();
    double expectedDetD = 0.0;

    if (detD != expectedDetD) {
        std::cerr << "Test failed for singular matrix: expected 0, got " << detD << '\n';
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

    if (detE != expectedDetE) {
        std::cerr << "Test failed for 4x4: expected " << expectedDetE << ", got " << detE << '\n';
        return 1;
    }

    // test identity matrix, should always be 1

    Matrix<double> I(4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            I(i, j) = (i == j) ? 1.0 : 0.0;

    double detI = I.determinant();
    if (detI != 1.0) {
        std::cerr << "Test failed for identity matrix: expected 1, got " << detI << '\n';
        return 1;
    }

    // test upper triangular matrix (det = product of diagonal)

    Matrix<double> T(3, 3);
    T(0,0)=2; T(0,1)=3; T(0,2)=1;
    T(1,0)=0; T(1,1)=5; T(1,2)=4;
    T(2,0)=0; T(2,1)=0; T(2,2)=7;

    double detT = T.determinant();
    double expectedDetT = 2 * 5 * 7; // 70

    if (detT != expectedDetT) {
        std::cerr << "Test failed for triangular matrix: expected " << expectedDetT << ", got " << detT << '\n';
        return 1;
    }

    return 0;
}