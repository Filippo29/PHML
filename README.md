# PHML

Starter layout for a modular C++ framework using CMake.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Run example

```bash
./build/examples/PHML_example_basic
```

## Install

```bash
cmake --install build --prefix ./install
```

## Directory overview

- `include/PHML/`: Public API headers.
- `src/`: Framework implementation.
- `tests/`: Unit and integration tests.
- `examples/`: Consumer-facing usage examples.
- `cmake/`: CMake helpers and toolchain files.
- `tools/`: Developer scripts.
- `.github/workflows/`: CI automation.

## Development guides

- Add a new framework module: `docs/adding-a-module.md`
