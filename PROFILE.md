## Instructions on kernel-level analysis with NVIDIA Nsight and PyTorch

### Notes:
1. the locations of nsys and nsight-cu-cli may vary from this guide
2. This guide assumes the user has setup a `script.py` to profile

### Steps
1. Setup Torch-addons for NCU: Use `torch.cuda.nvtx.range_push("start")`  and `torch.cuda.nvtx.range_pop()` around the region to profile.
2. Setup Torch-addons for NSYS: Use `torch.cuda.profiler.cudart().cudaProfilerStart()`  and `torch.cuda.profiler.cudart().cudaProfilerStop()` around the region to profile.
3. Allow for NSYS profiling: `sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'`
4. Profile with NCU: `sudo /opt/nvidia/nsight-compute/2021.2.0/nv-nsight-cu-cli -o output_ncu --set detailed --nvtx --nvtx-include "start/" python3 script.py`
5. Profile with NCU in CSV: `sudo /opt/nvidia/nsight-compute/2021.2.0/nv-nsight-cu-cli  --csv --set detailed --nvtx --nvtx-include "start/" python3 script.py  > output_ncu.csv`
6. Profile with NSYS: `nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o output_nsys --cudabacktrace=true --capture-range=cudaProfilerApi --stop-on-range-end=true  -f true -x true python3 script.py`
7. Convert NSYS output to CSV: `nsys stats --report gputrace --format csv,column --output .,- output_nsys.qdrep`
8. Refactor and analyze NCU script:
    * `python process_ncu.py <path to results directory>`
    * `python get_num_blocks.py <path to results directory>`
    * `python roofline_analysis.py <path to results directory>`
9. Generate plots: `python process_nsys.py`