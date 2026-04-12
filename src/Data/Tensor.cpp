#include "PHML/Data/Tensor.hpp"

#include <vector>
#include <stdexcept>

namespace PHML::Data {
    size_t Matrix::index(size_t i, size_t j) const {
        return i * cols + j;
    }

    Matrix::Matrix(size_t r, size_t c)
            : rows(r), cols(c), data(r * c, 0.0) {}
    Matrix::Matrix(size_t r, size_t c, double defaultVal)
            : rows(r), cols(c), data(r * c, defaultVal) {}

    double& Matrix::operator()(size_t i, size_t j) {
        return data[index(i, j)];
    }

    const double& Matrix::operator()(size_t i, size_t j) const {
        return data[index(i, j)];
    }

    Matrix Matrix::operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
}