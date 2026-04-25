#pragma once

#include "Core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

namespace PHML::Data {

    template <typename T>
    class Matrix : public Core<Matrix<T>> {
        using Base = Core<Matrix<T>>;

    public:

        // Constructors
        Matrix() = default;

        /// Allocate rows x cols, zero-initialised
        Matrix(std::size_t rows, std::size_t cols,
            Device dev = Device::cpu())
            : Base(dtype_of<T>(), dev)
            , rows_(rows), cols_(cols)
            , row_stride_(cols), col_stride_(1)
        {
            Base::init_storage(rows * cols * sizeof(T));
            std::fill(begin(), end(), T{0});
        }

        /// Allocate rows x cols, filled with default_val
        Matrix(std::size_t rows, std::size_t cols, T default_val,
            Device dev = Device::cpu())
            : Base(dtype_of<T>(), dev)
            , rows_(rows), cols_(cols)
            , row_stride_(cols), col_stride_(1)
        {
            Base::init_storage(rows * cols * sizeof(T));
            std::fill(begin(), end(), default_val);
        }

        /// Construct from flat initialiser list (row-major)
        Matrix(std::size_t rows, std::size_t cols,
            std::initializer_list<T> values,
            Device dev = Device::cpu())
            : Matrix(rows, cols, dev)
        {
            if (values.size() != rows * cols)
                throw std::invalid_argument(
                    "initializer_list size mismatch: expected " +
                    std::to_string(rows * cols) + ", got " +
                    std::to_string(values.size()));
            std::copy(values.begin(), values.end(), begin());
        }

        // Shapes
        std::size_t rows()       const { return rows_;        }
        std::size_t cols()       const { return cols_;        }
        std::size_t numel()      const { return rows_ * cols_; }
        std::size_t row_stride() const { return row_stride_;  }
        std::size_t col_stride() const { return col_stride_;  }
        bool        is_square()  const { return rows_ == cols_; }

        // Element access
        T& operator()(std::size_t r, std::size_t c) {
            bounds_check(r, c);
            return Base::template data_ptr<T>()[r * row_stride_ + c * col_stride_];
        }

        const T& operator()(std::size_t r, std::size_t c) const {
            bounds_check(r, c);
            return Base::template data_ptr<T>()[r * row_stride_ + c * col_stride_];
        }

        // operators

        Matrix<T> operator+(const Matrix<T>& other) const {
            shape_check(other, "operator+");
            Matrix<T> result(rows_, cols_, Base::device_);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < cols_; ++j)
                    result(i, j) = (*this)(i, j) + other(i, j);
            return result;
        }

        Matrix<T> operator-(const Matrix<T>& other) const {
            shape_check(other, "operator-");
            Matrix<T> result(rows_, cols_, Base::device_);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < cols_; ++j)
                    result(i, j) = (*this)(i, j) - other(i, j);
            return result;
        }

        /// Matrix multiplication (M x K) * (K x N) -> (M x N)
        Matrix<T> operator*(const Matrix<T>& other) const {
            if (cols_ != other.rows_)
                throw std::invalid_argument(
                    "operator*: incompatible shapes [" +
                    std::to_string(rows_) + "x" + std::to_string(cols_) +
                    "] * [" + std::to_string(other.rows_) + "x" +
                    std::to_string(other.cols_) + "]");
            Matrix<T> result(rows_, other.cols_, Base::device_);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t k = 0; k < cols_; ++k)
                    for (std::size_t j = 0; j < other.cols_; ++j)
                        result(i, j) += (*this)(i, k) * other(k, j);
            return result;
        }

        Matrix<T> operator*(T scalar) const {
            Matrix<T> result(rows_, cols_, Base::device_);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < cols_; ++j)
                    result(i, j) = (*this)(i, j) * scalar;
            return result;
        }

        friend Matrix<T> operator*(T scalar, const Matrix<T>& m) {
            return m * scalar;
        }

        // Linear algebra

        /// Zero-copy transpose — swaps strides, shares storage
        Matrix<T> transpose() const {
            Matrix<T> view;
            view.dtype_      = Base::dtype_;
            view.device_     = Base::device_;
            view.storage_    = Base::storage_;
            view.rows_       = cols_;
            view.cols_       = rows_;
            view.row_stride_ = col_stride_;
            view.col_stride_ = row_stride_;
            return view;
        }

        /// Determinant via LU decomposition with partial pivoting
        T determinant() const {
            square_check("determinant");
            Matrix<T> lu = deep_copy();
            const std::size_t n = rows_;
            T det = T{1};

            for (std::size_t k = 0; k < n; ++k) {
                std::size_t max_row = k;
                T max_val = std::abs(lu(k, k));
                for (std::size_t i = k + 1; i < n; ++i) {
                    T v = std::abs(lu(i, k));
                    if (v > max_val) { max_val = v; max_row = i; }
                }
                if (max_row != k) {
                    for (std::size_t j = 0; j < n; ++j)
                        std::swap(lu(k, j), lu(max_row, j));
                    det = -det;
                }
                if (std::abs(lu(k, k)) < std::numeric_limits<T>::epsilon())
                    return T{0};

                det *= lu(k, k);
                for (std::size_t i = k + 1; i < n; ++i) {
                    lu(i, k) /= lu(k, k);
                    for (std::size_t j = k + 1; j < n; ++j)
                        lu(i, j) -= lu(i, k) * lu(k, j);
                }
            }
            return det;
        }

        /// Inverse via Gauss-Jordan elimination
        Matrix<T> inverse() const {
            square_check("inverse");
            const std::size_t n = rows_;

            Matrix<T> aug(n, 2 * n, Base::device_);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j)
                    aug(i, j) = (*this)(i, j);
                aug(i, n + i) = T{1};
            }

            for (std::size_t col = 0; col < n; ++col) {
                std::size_t pivot = col;
                T best = std::abs(aug(col, col));
                for (std::size_t row = col + 1; row < n; ++row) {
                    T v = std::abs(aug(row, col));
                    if (v > best) { best = v; pivot = row; }
                }
                if (best < std::numeric_limits<T>::epsilon())
                    throw std::runtime_error("inverse: matrix is singular");
                if (pivot != col)
                    for (std::size_t j = 0; j < 2 * n; ++j)
                        std::swap(aug(col, j), aug(pivot, j));

                T pv = aug(col, col);
                for (std::size_t j = 0; j < 2 * n; ++j)
                    aug(col, j) /= pv;

                for (std::size_t row = 0; row < n; ++row) {
                    if (row == col) continue;
                    T factor = aug(row, col);
                    for (std::size_t j = 0; j < 2 * n; ++j)
                        aug(row, j) -= factor * aug(col, j);
                }
            }

            Matrix<T> result(n, n, Base::device_);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    result(i, j) = aug(i, n + j);
            return result;
        }

        // Static factories
        static Matrix<T> random(std::size_t r, std::size_t c,
                                T low = T{0}, T high = T{1},
                                Device dev = Device::cpu())
        {
            Matrix<T> m(r, c, dev);
            std::mt19937_64 rng{std::random_device{}()};
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dist(low, high);
                for (auto& v : m) v = dist(rng);
            } else {
                std::uniform_real_distribution<T> dist(low, high);
                for (auto& v : m) v = dist(rng);
            }
            return m;
        }

        static Matrix<T> identity(std::size_t n, Device dev = Device::cpu()) {
            Matrix<T> m(n, n, dev);
            for (std::size_t i = 0; i < n; ++i)
                m(i, i) = T{1};
            return m;
        }

        // Device transfer
        Matrix<T> to(Device target) const {
            if (Base::device_ == target) return *this;
            if (!target.is_cpu())
                throw std::runtime_error("CUDA transfer not yet implemented");
            return deep_copy(target);
        }

        // Iterators (range-for + std algorithms)
        T*       begin()       { return Base::template data_ptr<T>(); }
        T*       end()         { return begin() + numel(); }
        const T* begin() const { return Base::template data_ptr<T>(); }
        const T* end()   const { return begin() + numel(); }

        // Printing
        void print(std::ostream& os = std::cout) const {
            os << "Matrix<" << dtype_str(Base::dtype()) << "> ["
            << rows_ << "x" << cols_ << "] "
            << Base::device().str() << "\n";
            for (std::size_t r = 0; r < rows_; ++r) {
                os << "  [";
                for (std::size_t c = 0; c < cols_; ++c) {
                    os << (*this)(r, c);
                    if (c + 1 < cols_) os << ", ";
                }
                os << "]\n";
            }
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
            m.print(os); return os;
        }

    private:
        std::size_t rows_ = 0, cols_ = 0;
        std::size_t row_stride_ = 0, col_stride_ = 1;

        template <typename U>
        static DType dtype_of() {
            if constexpr (std::is_same_v<U, float>)    return DType::Float32;
            if constexpr (std::is_same_v<U, double>)   return DType::Float64;
            if constexpr (std::is_same_v<U, int32_t>)  return DType::Int32;
            if constexpr (std::is_same_v<U, int64_t>)  return DType::Int64;
            if constexpr (std::is_same_v<U, bool>)     return DType::Bool;
            throw std::runtime_error("Unsupported scalar type for Matrix");
        }

        void bounds_check(std::size_t r, std::size_t c) const {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range(
                    "Matrix(" + std::to_string(r) + "," +
                    std::to_string(c) + ") out of range [" +
                    std::to_string(rows_) + "x" + std::to_string(cols_) + "]");
        }

        void shape_check(const Matrix<T>& other, const char* op) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::invalid_argument(
                    std::string(op) + ": shape mismatch [" +
                    std::to_string(rows_) + "x" + std::to_string(cols_) +
                    "] vs [" + std::to_string(other.rows_) + "x" +
                    std::to_string(other.cols_) + "]");
        }

        void square_check(const char* op) const {
            if (!is_square())
                throw std::invalid_argument(
                    std::string(op) + ": matrix must be square, got [" +
                    std::to_string(rows_) + "x" + std::to_string(cols_) + "]");
        }

        Matrix<T> deep_copy(Device dev) const {
            Matrix<T> dst(rows_, cols_, dev);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < cols_; ++j)
                    dst(i, j) = (*this)(i, j);
            return dst;
        }

        Matrix<T> deep_copy() const { return deep_copy(Base::device_); }

        friend class Matrix<T>;
    };

    using AnyMatrix = std::variant<Matrix<float>, Matrix<double>,
                               Matrix<int32_t>, Matrix<int64_t>>;

    class MatrixFactory {
    public:
        MatrixFactory() = delete;
        MatrixFactory(const MatrixFactory&) = delete;
        MatrixFactory& operator=(const MatrixFactory&) = delete;

        static auto random(std::size_t r, std::size_t c, DType dt,
                    double low = 0.0, double high = 1.0,
                    Device dev = Device::cpu()) -> AnyMatrix
        {
            switch (dt) {
                case DType::Float32: return Matrix<float>::random(r, c, low, high, dev);
                case DType::Float64: return Matrix<double>::random(r, c, low, high, dev);
                case DType::Int32:   return Matrix<int32_t>::random(r, c, low, high, dev);
                case DType::Int64:   return Matrix<int64_t>::random(r, c, low, high, dev);
                default: throw std::runtime_error("Unsupported DType for random matrix");
            }
        }
    };

} // namespace PHML
