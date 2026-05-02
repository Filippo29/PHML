cmake_minimum_required(VERSION 3.21)

project(PHML VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(PHML_BUILD_TESTS "Build test targets" ON)
option(PHML_BUILD_EXAMPLES "Build example targets" ON)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(PHML_WITH_MPS_DEFAULT ON)
else()
    set(PHML_WITH_MPS_DEFAULT OFF)
endif()
option(PHML_WITH_MPS "Build with Apple MPS backend (Metal Performance Shaders)" ${PHML_WITH_MPS_DEFAULT})

add_library(PHML)
add_library(PHML::PHML ALIAS PHML)

set_target_properties(PHML PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_compile_features(PHML PUBLIC cxx_std_20)

target_include_directories(PHML
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(PHML_WITH_MPS)
    if(NOT APPLE)
        message(FATAL_ERROR "PHML_WITH_MPS requires an Apple platform")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp/Metal/Metal.hpp")
        message(FATAL_ERROR
            "PHML_WITH_MPS is ON but metal-cpp headers were not found at "
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp. "
            "Download metal-cpp from https://developer.apple.com/metal/cpp/ "
            "and extract it there, or configure with -DPHML_WITH_MPS=OFF.")
    endif()
    target_compile_definitions(PHML PUBLIC PHML_WITH_MPS)
    target_include_directories(PHML PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp)
    target_link_libraries(PHML PUBLIC
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework QuartzCore"
    )
endif()

add_subdirectory(src)

if(PHML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CTest)
if(BUILD_TESTING AND PHML_BUILD_TESTS)
    add_subdirectory(tests)
endif()

install(
    TARGETS PHML
    EXPORT PHMLTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT PHMLTargets
    FILE PHMLTargets.cmake
    NAMESPACE PHML::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PHML
)
