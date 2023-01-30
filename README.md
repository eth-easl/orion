# gpu_share

This repo contains the implementation of our scheduler for fine-grained GPU sharing across ML workloads.

### The current branch contains the implementation of CUDA/CPP backend. There are 3 main bugs that are currently being solved:
* inconsistent results of cuda capturing mechanism - differences with the Nsight Nsys trace
* seg fault at multi-threaded execution

Until memory problem is solved, we keep the scheduler as a 'referee' which blocks kernel execution accordingly.

### For CUDNN debugging:
* export CUDNN_LOGDEST_DBG=stdout
* export CUDNN_LOGINFO_DBG=1 

### Why do we see kernels being launched when printing tensors?

In PyTorch, the functions needed for printing tensors are defined in [this file](https://github.com/pytorch/pytorch/blob/master/torch/_tensor_str.py).
The class 'Formatter' calls specific torch functions, which lead to the launching of CUDA kernels, mainly referring to comparison, logical operations, indexing, abs, min/max, ceil, division, etc.
This is why we see kernels being launched when printing. This is verified with NVIDIA Nsight tool.
