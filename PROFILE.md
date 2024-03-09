## Instructions on kernel-level analysis with NVIDIA Nsight and PyTorch

### Notes:
1. the locations of nsys and nsight-cu-cli may vary from this guide
2. This guide assumes the user has setup a `script.py` to profile

### Profiling
1. Setup Torch-addons for NCU: Use `torch.cuda.nvtx.range_push("start")`  and `torch.cuda.nvtx.range_pop()` around the region to profile.
2. Setup Torch-addons for NSYS: Use `torch.cuda.profiler.cudart().cudaProfilerStart()`  and `torch.cuda.profiler.cudart().cudaProfilerStop()` around the region to profile.
3. Allow for NSYS profiling: `sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'`
4. Profile with NCU: `sudo /opt/nvidia/nsight-compute/2021.2.0/nv-nsight-cu-cli -o output_ncu --set detailed --nvtx --nvtx-include "start/" python3 script.py`
5. Profile with NCU in CSV: `sudo /opt/nvidia/nsight-compute/2021.2.0/nv-nsight-cu-cli  --csv --set detailed --nvtx --nvtx-include "start/" python3 script.py  > output_ncu.csv`
6. Profile with NSYS: `nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o output_nsys --cudabacktrace=true --capture-range=cudaProfilerApi --stop-on-range-end=true  -f true -x true python3 script.py`
7. Convert NSYS output to CSV: `nsys stats --report gputrace --format csv,column --output .,- output_nsys.qdrep`

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

Make sure the file is in a correct `.csv` format: depending on the NVIDIA CUDA version, and the type of profiling, the first line should look like that:

`"ID","Process ID","Process Name","Host Name","thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg","Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value","Rule Name","Rule Type","Rule Description"`


* `python profiling/postprocessing/get_num_blocks.py --results_dir <path to profiling files directory> --max_threads_sm <max_threads_per_sm> --max_blocks_sm <max_blocks_per_sm> --max_shmem_sm <max_shared_memory_per_sm> --max_regs_sm <max_registers_per_sm>`

You can find the maximum number of threads, blocks, shared memory and registers per SM in the GPU's architecture description.
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
* `python profiling/postprocessing/generate_file.py --input_file_name <path to the output_ncu_sms_roofline.csv file> --output_file_name <path to output file> --model_type <vision | bert | transformer>`
