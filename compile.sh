#!/bin/bash

cd src/cuda_capture && make libinttemp.so && cd ../../
cd src/scheduler && make scheduler_eval.so && cd ../../
