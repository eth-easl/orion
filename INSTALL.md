### Use Docker image

We have set up a docker image: [A]() with all packages pre-installed. We assume NVIDIA drivers are installed in the source machine, and that docker containers can use the host machine's GPUs.

* Start a container with `docker run --gpus=1 -it <A> bash`
* Download the Orion repo and install:
    * `git clone https://github.com/eth-easl/orion.git`
    * `cd orion/cpp_backend/scheduler && make scheduler_eval.so`

### Without Docker image

In order to use Orion without our pre-built image, a user must install:
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit). We have tested Orion with CUDA 10.2 and CUDA 11.3
* (optionally) [NVIDIA CUDNN](https://developer.nvidia.com/cudnn)
* Pytorch (from source)
* Download the Orion repo and install:
    * `git clone https://github.com/eth-easl/orion.git`
    * `cd orion/cpp_backend/scheduler && make scheduler_eval.so`