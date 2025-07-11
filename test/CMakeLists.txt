cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkTest)

set(CMAKE_CXX_STANDARD 17)

# Find the NeuralNetworkLib build (you need to build the lib first!)
link_directories(${CMAKE_SOURCE_DIR}/../build)   # Points to the library build output
include_directories(${CMAKE_SOURCE_DIR}/../src) # Expose library headers

add_executable(NeuralNetworkTest main.cpp)

# Link against NeuralNetworkLib (this assumes a static library was built)
target_link_libraries(NeuralNetworkTest PRIVATE NeuralNetworkLib)
