# Adding a New PHML Module

Use this checklist whenever you add another part of the framework.

## 1. Create the public API header

Create `include/PHML/io/logger.hpp`.

Example:

```cpp
#pragma once

#include <string>

namespace PHML::io {

    std::string format_message(const std::string& message);

} // namespace PHML::io
```

## 2. Create the implementation file

Create `src/io/logger.cpp`.

Example:

```cpp
#include "PHML/io/logger.hpp"

namespace PHML::io {

std::string format_message(const std::string& message) {
    return "[PHML] " + message;
}

} // namespace PHML::io
```

## 3. Register new source files in CMake

Update `src/CMakeLists.txt`:

```cmake
target_sources(PHML
    PRIVATE
        core/version.cpp
        io/logger.cpp
)
```

## 4. Expose the module in the umbrella header (optional)

Update `include/PHML/PHML.hpp`:

```cpp
#include "PHML/io/logger.hpp"
```

## 5. Add a unit test

Create `tests/unit/test_logger.cpp`.

Example:

```cpp
#include <cstdlib>

#include "PHML/io/logger.hpp"

int main() {
    return PHML::io::format_message("x") == "[PHML] x"
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
```

## 6. Register the test executable

Update `tests/CMakeLists.txt`:

```cmake
add_executable(test_logger unit/test_logger.cpp)
target_link_libraries(test_logger PRIVATE PHML::PHML)
add_test(NAME PHML_test_logger COMMAND test_logger)
```

## 7. Optionally add an example

If the module is user-facing, add `examples/<module>_usage.cpp` and register it in `examples/CMakeLists.txt`.

## Build and validate

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
