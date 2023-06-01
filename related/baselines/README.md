# TICK-TOCK scheduling

This directory contains a basic implementation of TICK-TOCK scheduling using Python threads, and torch.cuda streams and events.
It is based on the description provided in [WAVELET: EFFICIENT DNN TRAINING WITH TICK-TOCK SCHEDULING (MLSys'21)](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf).

What would be an interesting next step is implementing the memory management support described in [Zico: Efficient GPU Memory Sharing for
Concurrent DNN Training (ATC'21)](https://www.usenix.org/system/files/atc21-lim.pdf).

## Supported Models

### BERT

It is mainly adapted from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
(hereinafter referenced as "source") with the following difference:

1. For the lack of C++ enabled `apex` (which will be installed on Google Cloud soon), 
several places have been commented and marked with `TODO`.
2. All the jit related annotations, e.g. `@torch.jit.script` are commented.
4. Pretrained checkpoint is not loaded.

### dcgan

Mainly copied from https://github.com/pytorch/examples/blob/main/dcgan/main.py with no difference.

### gnmt
The differences with https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT
are:
1. We don't preallocate space (e.g. run a forward and backward iteration without updating weights) before starting a new epoch.
2. They set `pin_memory` as `True` while we don't.
3. All the jit related annotations are commented.

### retinanet
Differences with https://github.com/mlcommons/training/tree/master/single_stage_detector:
1. They set `pin_memory` as `True` while we don't.
2. I didn't set up the learning rate scheduler.
3. All the jit related annotations are commented.

### nasnet and vision
No difference

### transformer
Differences with https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL.
1. They explicitly disable profiling by the following, which I didn't do.
```python
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
```
2. They also do the following which is not done in other models:
```python
if 'apex' in sys.modules:
    amp.register_half_function(torch, 'einsum')
```
3. I didn't add the learning rate scheduler.
4. They set `pin_memory` as `True` while we don't.

## Guide to MPS

1. Execute `./start_MPS_control_daemon.sh` in one shell session
2. Exit that session and create a new one. Do
```shell
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
```
3Then start python program normally.
