export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.

export CUDA_MPS_PIPE_DIRECTORY=~/nvidia-mps # Select a location that’s accessible to the given $UID

export CUDA_MPS_LOG_DIRECTORY=~/nvidia-log # Select a location that’s accessible to the given $UID

nvidia-cuda-mps-control -d # Start the daemon.
