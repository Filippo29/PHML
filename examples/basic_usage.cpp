#include <iostream>

#include "PHML/PHML.hpp"
#include "PHML/Data/Tensor.hpp"

int main() {
    std::cout << "PHML version: "
              << PHML::core::version_string() << '\n';
    PHML::Data::Matrix<double> mat = PHML::Data::Matrix<double>::random(3, 3);
    mat.print();
    return 0;
}
