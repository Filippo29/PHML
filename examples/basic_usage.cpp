#include <iostream>

#include "PHML/PHML.hpp"

int main() {
    std::cout << "PHML version: "
              << PHML::core::version_string() << '\n';
    return 0;
}
