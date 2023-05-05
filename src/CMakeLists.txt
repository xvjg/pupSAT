# Set add all cpp and cuda files to SOURCES
file(GLOB SOURCES "*.cpp" "*.cu")

# Add -g flag for debug
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Get CUDA libraries
find_package(CUDA REQUIRED)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
