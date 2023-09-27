This is a simple example to check that Orion has been installed correctly and can run.

Please follow the instructions in [INSTALL](INSTALL.md) to start a container with our image.
Then start the Orion process (server and client) by running:
* `cd /root/orion/benchmarking`
* `LD_PRELOAD="/root/orion/src/cuda_capture/libinttemp.so" python launch_jobs.py /root/orion/artifact_evaluation/example/config.json 1 1 1`