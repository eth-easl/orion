# Introduction

We have set up a Google Cloud Platform (GCP) Project for VM creation and artifact evaluation.
We have created a GCP VM image with the NVIDIA drivers installed, to allow for faster deployment.
Please contact us with your GCP account to be added to the GCP project in order to conduct experiments.

We have set up a docker image: [fotstrt/orion-ae](https://hub.docker.com/repository/docker/fotstrt/orion-ae/general) with all packages pre-installed. We encourage reviewers to deploy and evaluate Orion using this image, as described in the [Artifact Evaluation section](#artifact-evaluation).


# Hardware Requirements

The artifact has been tested on a GCP VM with the following specifications:
* n1-standard-8 type (8 vCPUs, 30 GB DRAM)
* 1 V100-16GB GPU

# Software Requirements

* Ubuntu 18.04
* CMake 3.19
* CUDA 10.2
* CUDNN 7.6.5
* NVIDIA DRIVER version 510.47
* PyTorch 1.12 (installed from source, fully installed in the docker image)
* TorchVision 0.13
* Python >= 3.8
* BERT and Transformer-XL benchmarks from the [NVIDIA benchmarking repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling). (already contained in the docker image)



# Artifact Evaluation

Notes:
* We provide scripts for reproducing Figures 7 and 10.
* In order to reduce the GPU hours and cost of the experiments, we evaluate only ResNet50 and MobileNetV2 running as high-priority jobs in both cases, and compare Orion with the most competitive baselines (REEF and MPS), while also evaluating the ideal behavior.
* All experiments are repeated 3 times.

## Start a VM

We will need a machine with one V100-16GB GPU for the artifact evaluation.
We have set up a VM image with NVIDIA-DRIVERS preinstalled.
In order to create a VM, do

* `gcloud compute instances create <machine_name> --machine-type=n1-standard-8 --zone=europe-west4-a --boot-disk-size 500GB  --maintenance-policy TERMINATE --restart-on-failure --boot-disk-type pd-ssd --image image-nvidia-drivers --accelerator=count=1,type=nvidia-tesla-v100`

## SSH to the VM

After the VM is up and running, you can ssh by:

* `gcloud compute ssh <machine_name>`

## Start Orion container

* `docker pull fotstrt/orion-ae:v1`
* Start a container with `docker run --gpus=1 -it fotstrt/orion-ae:v1 bash`

If you encounter an issue like:

`Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/images/create?fromImage=fotstrt%2Forion-ae&tag=v1": dial unix /var/run/docker.sock: connect: permission denied`

please do

`sudo chmod 666 /var/run/docker.sock`


## Clone Orion repo and install

* `cd root && rm -rf orion`
* `git clone https://github.com/eth-easl/orion.git`
* `cd orion`
* `bash compile.sh`
* `pip install -e .`

## Run a simple example

* `LD_PRELOAD="/root/orion/src/cuda_capture/libinttemp.so" python benchmarking/launch_jobs.py --algo orion --config_file /root/orion/artifact_evaluation/example/config.json`

## Reproduce Fig 7

We assume we are at the `orion/artifact_evaluation/fig7` directory


### Create result directories

Run `bash prep_dirs.sh`
This will create a `results` directory, with sub-directories for the baselines that we will evaluate.

### Run ideal
In order to get the ideal p95 latency and/or throughput of the workloads, we will run them alone, without interference.

Run `python run_ideal.py`

This will populate the results under `fig7/results/ideal`.

### Run REEF

Run `python run_reef.py`

This will populate the results under `fig7/results/reef`.

### Run Orion

Run `python run_orion.py`

This will populate the results under `fig7/results/orion`.

### Run MPS

1. `bash ../../related/baselines/start_MPS_control_daemon.sh`
2. `cd config_files/mps`
3. `python run.py`
4. `cd ../..`
5. `bash ../../related/baselines/stop_MPS_control_daemon.sh`

This will populate the results under `fig7/results/mps`.

### Gather all results

1. `python gather_latency.py`
2. `python gather_throughput.py`

### Do plots (Fig 7a, 7b)

1. `python plot_latency.py`
2. `python plot_throughput.py`


The expected time for this experiment is 12 hours, and the expected cost is 16 USD in the proposed VM in GCP. See cost breakdown [here](https://cloud.google.com/products/calculator/#id=9b287eab-bcd6-496e-9ece-2116d69dd143)

## Reproduce Fig 10

We assume we are at the `orion/artifact_evaluation/fig10` directory

Run `bash prep_dirs.sh`
This will create a `results` directory, with sub-directories for the baselines that we will evaluate.

### Run ideal

Run `python run_ideal.py`

This will populate the results under `fig10/results/ideal`.

### Run REEF

Run `python run_reef.py`

This will populate the results under `fig10/results/reef`.

### Run Orion

Run `python run_orion.py`

This will populate the results under `fig10/results/orion`.

### Run MPS

1. `bash ../../related/baselines/start_MPS_control_daemon.sh`
2. `cd config_files/mps`
3. `python run.py`
4. `cd ../..`
5. `bash ../../related/baselines/stop_MPS_control_daemon.sh`

### Gather all results

Run `python gather_results.py`

### Do plot

Run `python plot_latency.py`

The expected time for this experiment is 8 hours, and the expected cost is 17 USD in the proposed VM in GCP.
See cost breakdown [here](https://cloud.google.com/products/calculator/#id=9b287eab-bcd6-496e-9ece-2116d69dd143).