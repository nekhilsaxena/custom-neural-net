cmake_minimum_required(VERSION 3.15)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/Model
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Network
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Neuron
)

# Source files for the main target
set(SOURCES
    src/main.cpp
    src/Model/Model.cpp
    src/Model/Components/Network/Network.cpp
    src/Model/Components/Layer/Layer.cpp
    src/Model/Components/Neuron/Neuron.cpp
)

add_executable(NeuralNetwork ${SOURCES})

# Add test subdirectory
add_subdirectory(test)
