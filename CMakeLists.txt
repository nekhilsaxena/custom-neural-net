cmake_minimum_required(VERSION 3.15)
project(NeuralNetworkLib)

set(CMAKE_CXX_STANDARD 17)

# Collect all source files for the library
set(LIB_SOURCES
    src/Model/Model.cpp
    src/Model/Components/Network/Network.cpp
    src/Model/Components/Network/NetworkUtils.cpp
    src/Model/Components/Layer/Dense/Dense.cpp
    src/Model/Components/Layer/Dropout/Dropout.cpp
    src/Model/Components/Neuron/Neuron.cpp
)

# Create a STATIC library
add_library(NeuralNetworkLib STATIC ${LIB_SOURCES})

# Public include directories
target_include_directories(NeuralNetworkLib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/Model
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Network
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer/Dense
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Layer/Dropout
    ${CMAKE_SOURCE_DIR}/src/Model/Components/Neuron
)

# (Optional) Install step to make the library available elsewhere
install(TARGETS NeuralNetworkLib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h")
