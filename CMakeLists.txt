project(ParallemProgrammingFinal LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.18)

# find all source files
file(GLOB SOURCES src/*.cu src/*.cpp)

# set DATA_DIR macro
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
configure_file(
    ${CMAKE_SOURCE_DIR}/config.h.in
    ${CMAKE_BINARY_DIR}/config.h
)

# make compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# make executable
set(CMAKE_CUDA_ARCHITECTURES 61 75)
add_executable(extract_features ${SOURCES})

target_include_directories(
	extract_features 
	PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}   # for config.h
)