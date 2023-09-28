We have set up a Google Cloud Platform (GCP) Project for VM creation and artifact evaluation.
Please contact us with your GCP account to be added to the GCP project in order to conduct experiments.

## Start a VM

We will need a machine with one V100-16GB GPU for the artifact evaluation.
We have set up a VM image with NVIDIA-DRIVERS preinstalled.
In order to create a VM, do

* `gcloud compute instances create <machine_name> --machine-type=n1-standard-8 --zone=europe-west4-a --boot-disk-size 500GB  --maintenance-policy TERMINATE --restart-on-failure --boot-disk-type pd-ssd --image image-nvidia-drivers --accelerator=count=1,type=nvidia-tesla-v100`

## SSH to the VM

After the VM is up and running, you can ssh by:

* `gcloud compute ssh <machine_name>`

## Clone Orion repo and install

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

### Run Orion

### Run MPS

### Gather all results

### Do plots (Fig 7a, 7b)

The expected time for this experiment is X hours, and the expected cost is Y USD

## Reproduce Fig 8

### Run ideal

### Run REEF

### Run Orion

### Run MPS

### Gather all results

### Do plot

The expected time for this experiment is X hours, and the expected cost is Y USD