#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

__global__ void ksleep(int64_t num_cycles);
extern "C" void sleep_kernel(int64_t num_cycles, cudaStream_t stream_id);