# custom-neural-network

A C++ neural network project with example datasets, visualization tools, and native library integrations. This README documents repository structure, build instructions, usage examples, API overview, testing, CI suggestions, and contribution guidelines. The project includes ImGui-based visualization and vcpkg-managed native dependencies.

## Project Overview

This repository implements a small, self-contained neural network framework in modern C++. It includes:

- Dataset CSVs and example data used for training and evaluation.
- Core training, inference, and utility code under src/.
- Example applications and ImGui-based visualizers in examples/.
- vcpkg metadata and installed packages for reproducible native dependency management.

This fork includes customizations and additions by the project author (repository import indicates the author). These changes include dataset additions, example integrations, and project packaging for vcpkg-based workflows.

## Key Features

A short list of implemented features and what this fork adds:

- Feedforward neural network training loop with configurable layers, activations, and learning rate.
- Data loading utilities for CSV/structured example data included in data/.
- Real-time UI for monitoring training progress and visualizing weights using Dear ImGui.
- CMake-based build with optional vcpkg toolchain file for dependency resolution.

## Repository Layout

Top-level layout and purpose of key folders and files.

data/
Example CSVs and training/test datasets used by example apps.

src/
Core neural network implementation, utilities, and any custom extensions added by the author.

examples/
Example binaries that load datasets and run training sessions with ImGui visualizations.

CMakeLists.txt
Root CMake configuration that finds libraries and defines targets.

vcpkg_installed/
vcpkg-generated metadata and CMake import files for glad, nlohmann-json, and others.

## Requirements

Short summary of development and runtime requirements:

- C++17 compatible compiler (GCC, Clang, MSVC).
- CMake 3.15+ recommended.
- vcpkg for native dependency management (recommended).
- Optional: GPU/OpenGL drivers if you use the ImGui/OpenGL visualizers.

## Build Instructions

This section shows common build flows. Each snippet is copy-paste ready.

1) Using vcpkg toolchain (recommended)

## Clone repo
git clone <repo-url> neural-network
cd neural-network

## Create build dir
mkdir build && cd build

## Configure with vcpkg toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ..
cmake --build . --config Release


Notes: 
- Ensure required vcpkg packages are installed (nlohmann-json, glad, imgui, stb, etc.).
- vcpkg metadata is present in the repository for reference.

2) Native CMake (without vcpkg)

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

If using system-installed libraries, ensure include paths and link flags satisfy CMake find_package calls.

## Running Examples

After building, example binaries are emitted under build/bin or build/examples depending on CMake setup. Example usage:

## Example training run
./build/examples/example_train --data ../data/sample_dataset.csv --epochs 200 --lr 0.01

## Launch ImGui visualizer (if built)
./build/examples/visualizer --model ./models/latest.nn


Examples show ImGui initialization and callback wiring in the codebase. See the examples/ folder for exact command-line flags and usage.

## Configuration and Command-line Flags

Common command-line flags supported by examples (adapted from example code in this repo):

- --data : Path to CSV dataset.
- --epochs : Number of training epochs.
- --lr : Learning rate.
- --batch : Batch size for training.
- --seed : RNG seed.

Check example binary help output for the exact list of supported flags.

## API Overview

Short description of major classes and functions implemented in src/.

NN::Network
Represents a feedforward neural network. Methods: addLayer, forward, backward, updateWeights, save, load.

NN::Layer (Dense, Activation)
Dense layer supports weight initialization, bias, and per-layer learning hyperparameters. Activation layers provide ReLU, Sigmoid, Tanh, Softmax implementations.

Data::CSVLoader
Loads CSV files into numeric matrices, handles header and normalization options.

Viz::ImGuiMonitor
ImGui-based monitor that visualizes loss curves, weight histograms, and sample predictions in real time.

Example usage snippet (minimal):

```
#include "Network.h"
#include "CSVLoader.h"

int main() {
auto data = Data::CSVLoader::load("data/sample.csv");

text
NN::Network net;
net.addLayer(NN::Dense( input_size, 64 ));
net.addLayer(NN::Activation::ReLU());
net.addLayer(NN::Dense(64, output_size));

net.train(data, epochs, learning_rate);
net.save("models/my_model.nn");
```

Adjust symbols and namespaces to match actual headers in src/.

## Tests

Unit tests (if present) are under tests/ or use a test target in CMake. Run with ctest from the build directory:

Adjust symbols and namespaces to match actual headers in src/.

## Tests

Unit tests (if present) are under tests/ or use a test target in CMake. Run with ctest from the build directory:

```bash
cd build
ctest --output-on-failure
```

Model files are simple binary or JSON files saved by Network::save. Use reproducible names with model metadata (hyperparameters, epoch, date).

## Code Ownership and Customizations

This repository import and the customizations are from the project author. Notable author-specific changes include:

- Added example datasets and CSV formatting conventions used by the training scripts.
- Integrated Dear ImGui visualizers and example backends for monitoring training.
- Packaged repository for vcpkg integration and included generated vcpkg metadata to ease dependency resolution.

If you want the README to call out specific source files you modified, tell me which filenames or paste the function/class names you added or changed. I can add a dedicated "What I changed" diff-style section.

## Contributing

- Fork the repo and create a feature branch.
- Run and add tests for new features.
- Follow the repository's code style and include comments for complex logic.
- Open a pull request describing the change and include reproducible steps.

## Troubleshooting

- If ImGui examples fail to find an OpenGL context, ensure your system has proper OpenGL drivers and that glad is correctly linked. ImGui backend initialization examples exist in the examples folder.
- If CMake cannot find vcpkg packages, ensure the toolchain file path is correct and the packages are installed via vcpkg. Metadata and import files are present for reference.

## Third-party Licenses and Acknowledgements

Third-party components used in this repository include:

- Dear ImGui — UI toolkit used by examples.
- glad — OpenGL loader; vcpkg import files are present.
- nlohmann-json — JSON library; usage metadata and license text available in vcpkg metadata.
- stb — Image loading library.
- glm — Mathematics library.

Third-party license texts are included in vendor or vcpkg metadata files.

## License

See LICENSE in the repository root for licensing terms. Some bundled third-party components carry their own licenses (e.g., MIT for nlohmann-json).


