## Instructions on kernel-level analysis with NVIDIA Nsight and PyTorch

### Notes:
1. the locations of nsys and ncu may vary from this guide
2. This guide assumes the user has setup a `script.py` to profile
3. nsys/ncu flags might slightly vary depending on the version you are using.

### Profiling
1. Setup Torch-addons for profiling only a specific part of the code: Use `torch.cuda.profiler.cudart().cudaProfilerStart()`  and `torch.cuda.profiler.cudart().cudaProfilerStop()` around the region to profile.
2. Profile with NCU: `ncu -o output_ncu --set detailed --profile-from-start off -f python3 script.py`
5. Profile with NCU in CSV: `ncu  --csv --set detailed --profile-from-start off python3 script.py  > output_ncu.csv`
6. Profile with NSYS: `nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o profiles/output_nsys --cudabacktrace=true --capture-range=cudaProfilerApi  -f true -x true python3 script.py`
7. Convert NSYS output to CSV: `nsys stats --report cuda_gpu_trace --format csv,column --output .,- output_nsys.nsys-rep`

At this point, 4 files should have been generated:
* `output_ncu.ncu-rep`
* `output_ncu.csv`
* `output_nsys.qdrep`
* `output_nsys_gputrace.csv`

Using Nsight Compute, open the `output_ncu.ncu-rep` file, and download the raw csv file as `raw_ncu.csv`.


### Extracting resource utilization info
Extract the required information from the profiling files:
* `python profiling/postprocessing/process_ncu.py --results_dir <path to profiling files directory>`

If the `output_ncu.csv` file contains any program logs that do not conform with the `.csv` format, this command might throw errors.

* `python profiling/postprocessing/get_num_blocks.py --results_dir <path to profiling files directory> --max_threads_sm <max_threads_per_sm> --max_shmem_sm <max_shared_memory_per_sm> --max_regs_sm <max_registers_per_sm>`

You can find the maximum number of threads, blocks, shared memory and registers per SM in the GPU's architecture description (or from NCU).
By default, the `get_num_blocks.py` is configured for the [NVIDIA Tesla V100 GPU](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).

* `python profiling/postprocessing/roofline_analysis.py --results_dir <path to profiling files directory> --ai_threshold <ai_threshold>`

Note that `ai_threshold` stands for the 'knee' arithmetic intensity of the roofline plot taken from the Nsight Compute tool, and might be different for each GPU.

After these steps, an `output_ncu_sms_roofline.csv` should have been generated.

### (Optional) Plot traces
You can use the  `profiling/postprocessing/process_nsys.py` file to generate resource utilization plot traces over time.
* `python profiling/postprocessing/process_nsys.py --results_dir <path to profiling files directory> --max_sms <max SMs in the GPU> --metric <SM | Comp | Mem>`

### Postprocessing to convert to a kernel info file for Orion to use
This reads the profiling file and keeps the necessary information needed for each kernel (Number of SMs, Profile, Duration).
It also groups kernels into operators, e.g. if a CUDNN Convolution operator has 2 kernels, it will group them into one operator.
* `python profiling/postprocessing/generate_file_updated.py --input_file_name <path to the output_ncu_sms_roofline.csv file> --output_file_name <path to output file> --model_type <vision | bert | transformer>`
