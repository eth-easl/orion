# gpu_share

This repo contains the implementation of our scheduler for fine-grained GPU sharing across ML workloads.
The two basic instances are the intercepter (CUDA and CUDNN calls) and the Scheduler.

### How to run?

Currently, the scheduler requires torch being installed from source, and the source code being located at $HOME/pytorch.
The file 'launch_jobs.py' is responsible for spawning the scheduler and the application thread(s).

### For CUDNN debugging:
* export CUDNN_LOGDEST_DBG=stdout
* export CUDNN_LOGINFO_DBG=1 

### For CUBLAS debugging:
* export CUBLAS_LOGDEST_DBG=stdout
* export CUBLAS_LOGINFO_DBG=1

### Why do we see kernels being launched when printing tensors?

In PyTorch, the functions needed for printing tensors are defined in [this file](https://github.com/pytorch/pytorch/blob/master/torch/_tensor_str.py).
The class 'Formatter' calls specific torch functions, which lead to the launching of CUDA kernels, mainly referring to comparison, logical operations, indexing, abs, min/max, ceil, division, etc.
This is why we see kernels being launched when printing. This is verified with NVIDIA Nsight tool.
