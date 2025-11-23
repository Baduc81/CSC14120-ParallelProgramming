# CSC14120 Parallel Programming Final Project

Feature extraction from CIFAR-10 dataset using CUDA for parallel computing.

## Introduction

This project implements image feature extraction algorithms using GPU acceleration with CUDA to leverage parallel computing capabilities on the CIFAR-10 dataset.

## Requirements

- **CUDA Toolkit**: 11.0+ 
- **CMake**: 3.18+
- **C++ Compiler**: Supporting C++11 or later
- **GPU**: NVIDIA GPU with Compute Capability 6.1+ (configured for sm_61, sm_75)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Ddyln/CSC14120-ParallelProgramming.git
cd CSC14120-ParallelProgramming
```

### 2. Prepare the data
First, create a `data/` folder.
Then install the CIFAR-10 dataset **binary version** from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and place it inside the `data/` directory. Then extract it.

## Run the program
### Features extraction phase
You can use the script that I have prepared:
```bash
chmod +x ./scripts/run_extract_features.sh
./scripts/run_extract_features.sh
```

or to do it manually:
```bash
cmake -S . -B build
cmake --build build

# To run the extrace features phase, use the following command
# from the project root directory
./build/extract_features
```


### Train SVM phase

## 

## Members
| Student ID     | Name       |
|-|-|
|22120071		 | Phan Bá Đức|
|22120078		 | Nguyễn Thành Đức|
|22120182		 | Đặng Duy Lân| 

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CMake Documentation](https://cmake.org/documentation/)
