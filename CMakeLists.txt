cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)

# Tell CMake where vcpkg is
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# Add your source files
file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.h
)

add_executable(NeuralNetwork ${SOURCES})

# Required packages
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# Link packages to your project
target_link_libraries(NeuralNetwork
    PRIVATE
        imgui::imgui
        implot::implot
        glfw
        opengl32
        glad::glad
)
