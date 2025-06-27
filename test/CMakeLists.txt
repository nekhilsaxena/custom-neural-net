cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkTest)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../src/Model
    ${CMAKE_SOURCE_DIR}/../src/Model/Components/Network
    # ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer/Dense
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer/Dropout
    ${CMAKE_SOURCE_DIR}/../src/Model/Components/Neuron
)

# Source files
set(SOURCES
    main.cpp
    ../src/Model/Model.cpp
    ../src/Model/Components/Network/Network.cpp
    ../src/Model/Components/Layer/Dense/Dense.cpp
    ../src/Model/Components/Layer/Dropout/Dropout.cpp
    ../src/Model/Components/Neuron/Neuron.cpp
)

add_executable(NeuralNetworkTest ${SOURCES})
