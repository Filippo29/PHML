# PHML

Starter layout for a modular C++ framework using CMake.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

On **Apple Silicon** (M1/M2/M3/M4), the Metal/MPS GPU backend is enabled automatically. See [Apple Silicon / MPS build](#apple-silicon--mps-build) below.

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

---

## Apple Silicon / MPS build

PHML includes a Metal compute backend for `PHML::Data` that dispatches `float32` matrix and tensor operations to the Apple GPU via Metal Shading Language kernels.

### Prerequisites

1. **Apple Silicon Mac** (M-series). The backend uses `MTLStorageModeShared` (unified memory) and is not supported on Intel Macs.
2. **Xcode command-line tools** — `xcode-select --install`.
3. **metal-cpp** (Apple's header-only C++ wrapper for Metal). Download the latest release from [developer.apple.com/metal/cpp](https://developer.apple.com/metal/cpp/) and extract it so the layout is:
   ```
   third_party/metal-cpp/
   ├── Metal/
   │   └── Metal.hpp
   ├── Foundation/
   │   └── Foundation.hpp
   └── QuartzCore/
       └── QuartzCore.hpp
   ```

### Configure and build

```bash
# MPS on (default on arm64)
cmake -S . -B build -DPHML_WITH_MPS=ON
cmake --build build -j

# MPS off (e.g. for Intel, CI, or a CPU-only build on Apple Silicon)
cmake -S . -B build -DPHML_WITH_MPS=OFF
cmake --build build -j
```

If metal-cpp is not found and `PHML_WITH_MPS=ON`, CMake will stop with a clear error pointing to the expected path.

### What the backend accelerates

| Operation | dtype | Execution |
|---|---|---|
| Matrix multiplication | `float32` | GPU (Metal compute kernel) |
| Element-wise `+`, `-`, `*` (Hadamard), scalar `*` | `float32` | GPU (Metal compute kernel) |
| All operations | `float64`, `int32`, `int64` | CPU via shared-memory pointer (one-time notice printed) |
| Non-contiguous layouts (after `transpose`, `permute`) | any | CPU |

`double` and integer operations fall back to the CPU automatically using the `MTLBuffer`'s CPU-accessible pointer — no data copy is needed. A one-time message is printed to `stderr` the first time each `(op, dtype)` pair hits the fallback:

```
[PHML::MPS] 'matrix::operator*' has no GPU implementation for dtype 'float64';
running on CPU via shared-memory pointer (this notice is shown once).
```

### Verifying GPU dispatch

Run the bundled example to see CPU vs. MPS timing for both `double` (fallback) and `float` (GPU):

```bash
./build/examples/PHML_example_basic
```

Expected output shape:

```
=== Matrix<double> 256x256 ===
  CPU :  8412 µs
  MPS :  7980 µs  (fallback — no GPU kernel for float64)

=== Matrix<float> 256x256 ===
  CPU :  7105 µs
  MPS :   430 µs  (GPU dispatch)
  Speedup: 16.5x
```

Speedup varies by matrix size and chip generation; it grows significantly for larger matrices.

### Backend internals

See [`include/PHML/Data/Backend/README.md`](include/PHML/Data/Backend/README.md) for a detailed description of the architecture, shader kernels, thread safety, and how to extend the backend with new operations.

---

## Directory overview

- `include/PHML/`: Public API headers.
  - `Data/Backend/`: Metal/MPS GPU backend headers (compiled only with `PHML_WITH_MPS=ON`).
- `src/`: Framework implementation.
  - `Data/Backend/`: Metal/MPS GPU backend sources.
- `tests/`: Unit and integration tests.
- `examples/`: Consumer-facing usage examples.
- `cmake/`: CMake helpers and toolchain files.
- `third_party/`: Vendored dependencies (metal-cpp goes here).
- `tools/`: Developer scripts.
- `.github/workflows/`: CI automation.

## Development guides

- Add a new framework module: `docs/adding-a-module.md`
