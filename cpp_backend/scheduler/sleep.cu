#include "sleep.h"

__global__ void ksleep(int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock();
    while(cycles < num_cycles) {
        cycles = clock() - start;
    }
}

extern "C" void sleep_kernel(int64_t num_cycles, cudaStream_t stream)
{
    // Our kernel will launch a single thread to sleep the kernel
    int blockSize, gridSize;
    blockSize = 1;
    gridSize = 1;

    // Execute the kernel
    ksleep<<<gridSize, blockSize, 0, stream>>>(num_cycles);
}