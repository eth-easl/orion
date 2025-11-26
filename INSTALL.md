The following guide provides instructions on how to use Orion inside a docker container:

1. Build a docker image using the [Dockerfile](setup/Dockerfile)
2. Start the container with access to a gpu
3. Clone, compile and install Orion:
* `git clone https://github.com/eth-easl/orion.git`
* `cd orion`
* `bash compile.sh`
* `pip install -e .`