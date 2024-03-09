# GPU Sharing Baselines
This directory contains evaluations of GPU sharing techniques between two workloads.
Supported baselines are `MPS`, `TickTock`, `Streams`, `Isolated`, and `Sequential`.

[main.py](./main.py) is the entry point of the evaluation and the all configurations are in [config.yaml](./config.yaml).

To evaluate a baseline, change the `policy` field in `config.yaml` to the baseline name.
Then, run `python main.py --config config.yaml`.

If no `--config` argument is provided, [config.yaml](./config.yaml) is used by default.


## Supported Baselines
### MPS
MPS: [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) is a feature of NVIDIA GPUs that allows multiple processes to share a single GPU.

**Caveat!** There are extra steps to do before executing the python program:
1. Execute `./start_MPS_control_daemon.sh` to start the MPS server.
2. Export these two environment variables:
```shell
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
```
3. Within the same shell session where you exported the environment variables, execute the python program normally.

### TICK-TOCK scheduling

This directory contains a basic implementation of TICK-TOCK scheduling using Python threads, and torch.cuda streams and events.
It is based on the description provided in [WAVELET: EFFICIENT DNN TRAINING WITH TICK-TOCK SCHEDULING (MLSys'21)](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf).

What would be an interesting next step is implementing the memory management support described in [Zico: Efficient GPU Memory Sharing for
Concurrent DNN Training (ATC'21)](https://www.usenix.org/system/files/atc21-lim.pdf).

### Streams
GPU Streams provide a way to execute workloads concurrently on a single GPU.
One stream captures a linear sequence of operations to be executed, and multiple streams can be executed concurrently.

### Sequential
`Sequential` represents the temporal sharing baseline where the GPU is time-sliced between the two workloads.

### Isolated
To analyze the overhead of GPU sharing, we compare the performance of GPU sharing with the performance of executing 
the workload on a single GPU without sharing. For `Isolated` we first execute workload A and then workload B after A is finished.
