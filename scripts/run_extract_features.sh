#!/bin/bash
set -ex

input_folder="$1"
output_folder="$2"

if [ -d "build" ]; then
	rm -rf build
fi
cmake -S . -B build
cmake --build build
./build/extract_features "$input_folder" "$output_folder"