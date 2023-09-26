# Orion

Orion is a fine-grained scheduler for interference-free GPU sharing across ML workloads.

## Table of Contents
- [Introduction](#introduction)
- [Example](#example)
- [Project Structure](#project-structure)
- [Hardware Requirement](#hardware-requirement)
- [Installation](#installation)
- [Debugging](#debugging)
- [Paper](#paper)

## Introduction

Orion is a fine-grained, interference-free scheduler for GPU sharing across ML workloads. We assume one of the clients is high-priority, while the rest of the clients are best-effort.

Orion intercepts CUDA, CUDNN, and CUBLAS calls and submits them into software queues.
The _Scheduler_ polls these queues and schedules operations based on their resource requirements and their priority. See [ARCHITECTURE](ARCHITECTURE.md) for more details on the system and the scheduling policy.

Orion expects that each submitted job has a file where all of its operations, along with their profiles and Straming Multiprocessor (SM) requirements are listed. See [PROFILE](PROFILE.md) for detailed instructions on how to profile a client applications, and how to generate the profile files.

## Example

Follow the instructions on [INSTALL](INSTALL.md), to install Orion and its dependencies, or launch a Docker container with Orion preinstalled.

See [PROFILING](PROFILING.md) to generate profiling files for each workload.
Create a json file containing all the info for the workloads that are about to share the GPU. See examples under 'eval'.

The file 'launch_jobs.py' is responsible for spawning the scheduler and the application thread(s).
`python launch_jobs.py <config file> <orion kernel time limit> <hp_limit> <update start>`

## Project Structure
```
> tree .
├── profiling                     # Scripts and instructions for profiling
│   ├── benchmarks                # Scripts of DNN models for profiling
│   ├── postprocessing            # Scripts for processing of profile files
└── src                           # Source code
│   ├── cuda_capture              # Code to intercept CUDA/CUDNN/CUBLAS calls
│   └── scheduler                 # Implementation of the scheduling policy
│   └── scheduler_frontend.py     # Python interface for the Orion scheduler
└── benchmarking_suite            # Scripts and configuration files for benchmarking
└── artifact_evaluation           # Scripts and instructions for artifact evaluation
```

## Hardware Requirements
Orion currently supports NVIDIA GPUs.

## Installation
see [INSTALL](INSTALL.md).

## Debugging
see [DEBUGGING](DEBUGGING.md).

## Paper
If you use Orion, please cite our paper: (TODO)
