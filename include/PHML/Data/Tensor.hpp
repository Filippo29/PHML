#pragma once

#include <vector>

namespace PHML::Data {
    class Matrix {
    private:
        size_t rows, cols;
        std::vector<double> data;

        size_t index(size_t i, size_t j) const;

    public:
        Matrix(size_t r, size_t c); // Constructor with default value 0.0
        Matrix(size_t r, size_t c, double defaultVal);
        void print() const;

        double& operator()(size_t i, size_t j);
        const double& operator()(size_t i, size_t j) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;

        Matrix transpose() const;
        Matrix inverse() const;

        static Matrix random(size_t r, size_t c);
    };
}