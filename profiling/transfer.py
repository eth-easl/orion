import os

files = ["output_ncu.ncu-rep", "output_ncu.csv", "output_nsys.qdrep", "output_nsys_gputrace.csv"]
for file in files:
    cmd = f"gcloud compute scp image-varuna@vm-torch-src-1:/home/image-varuna/gpu_share_repo/profiling/benchmarks/{file} traces/vision/resnet50_32_fwd/ --project ml-elasticity"
    os.system(cmd)