#include <cuda_runtime.h>
#include <stdio.h>

#include "common/gpu_info.h"

void gpu_info::print() {
    int device;
    cudaGetDevice(&device);
    printf("Using device: %d\n", device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("1. GPU card's name: %s\n", prop.name);
    printf("2. GPU computation capabilities: %d.%d\n", prop.major, prop.minor);
    printf(
        "3. Maximum number of block dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0],
        prop.maxThreadsDim[1],
        prop.maxThreadsDim[2]
    );
    printf(
        "4. Maximum number of grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0],
        prop.maxGridSize[1],
        prop.maxGridSize[2]
    );
    printf(
        "5. Maximum size of GPU memory: %.2f GB\n",
        (double)prop.totalGlobalMem / (1 << 30)
    );
    printf(
        "6. Amount of constant memory: %.2f KB\n",
        (double)prop.totalConstMem / 1024.0
    );
    printf(
        "   Amount of shared memory per block: %.2f KB\n",
        (double)prop.sharedMemPerBlock / 1024.0
    );
    printf("7. Warp size: %d\n", prop.warpSize);
}

void gpu_info::print_memory_info() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    double used_mb = (double)(total_mem - free_mem) / (1 << 20);
    double total_mb = (double)total_mem / (1 << 20);
    printf("GPU Memory: %.2f / %.2f MB used\n", used_mb, total_mb);
}

double gpu_info::get_gpu_memory_used_gb() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return (double)(total_mem - free_mem) / (1LL << 30);
}