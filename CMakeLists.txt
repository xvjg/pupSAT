# Project: GPU SAT Solver
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(purrSAT)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Put binaries in the root build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Subdirectories
add_subdirectory(src)
