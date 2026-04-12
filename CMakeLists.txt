cmake_minimum_required(VERSION 3.21)

project(PHML VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(PHML_BUILD_TESTS "Build test targets" ON)
option(PHML_BUILD_EXAMPLES "Build example targets" ON)

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
