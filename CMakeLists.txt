project(ParallemProgrammingFinal LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.18)

# Enable optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# find all source files
file(GLOB SOURCES src/*.cu src/*.cpp)

# make compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# make executable
set(CMAKE_CUDA_ARCHITECTURES 61 75)
add_executable(extract_features ${SOURCES})

target_include_directories(
	extract_features 
	PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)