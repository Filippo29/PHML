#include <iostream>
#include <chrono>

#include "PHML/PHML.hpp"
#include "PHML/Data/Matrix.hpp"
#include "PHML/Data/Tensor.hpp"

int main() {
    std::cout << "PHML version: "
              << PHML::core::version_string() << '\n';
    PHML::Data::AnyMatrix mat1 = PHML::Data::MatrixFactory::random(100, 100, PHML::Data::DType::Float64);
    PHML::Data::AnyMatrix mat2 = PHML::Data::MatrixFactory::random(100, 100, PHML::Data::DType::Float64);
    auto start = std::chrono::high_resolution_clock::now();
    PHML::Data::Matrix<double> result = std::get<PHML::Data::Matrix<double>>(mat1) * std::get<PHML::Data::Matrix<double>>(mat2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Computation time: " << duration.count() << " microseconds" << '\n';

    PHML::Data::Tensor<double> tensor = PHML::Data::Tensor<double>::random({2, 2, 3});
    std::cout << "Tensor:\n" << tensor << '\n';
    std::cout << "Tensor transpose:\n" << tensor.reshape({1, 3, 4}) << '\n';
    std::cout << "Tensor transpose:\n" << tensor.transpose(0, 2) << '\n';
    

    return 0;
}
