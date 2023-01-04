# gpu_share

This repo contains the implementation of our scheduler for fine-grained GPU sharing across ML workloads.

### The current branch contains the implementation of CUDA/CPP backend. There are 3 main bugs that are currently being solved:
* inconsistent results of cuda capturing mechanism - differences with the Nsight Nsys trace
* seg fault at multi-threaded execution, even when printing!
* multiple kernels when printing - why?
