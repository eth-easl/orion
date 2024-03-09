### Basic library to capture CUDA calls

This captures only cudaLaunchKernel and cudaMalloc for now. It is also applicable for PyTorch programs.

### Compile

make all

### Run

LD_PRELOAD="<full path to the library" <my_program>

