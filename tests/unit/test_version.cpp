#include <cstdlib>
#include <iostream>

#include "PHML/core/version.hpp"

int main() {
    if (PHML::core::version_major() != 0) {
        std::cerr << "Expected major version 0\n";
        return EXIT_FAILURE;
    }

    if (PHML::core::version_minor() != 1) {
        std::cerr << "Expected minor version 1\n";
        return EXIT_FAILURE;
    }

    if (PHML::core::version_patch() != 0) {
        std::cerr << "Expected patch version 0\n";
        return EXIT_FAILURE;
    }

    if (PHML::core::version_string() != "0.1.0") {
        std::cerr << "Expected version string 0.1.0\n";
        return EXIT_FAILURE;
    }

    std::cout << "All tests passed\n";
    return EXIT_SUCCESS;
}
