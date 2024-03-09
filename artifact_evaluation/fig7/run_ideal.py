import os
import time

num_runs = 3
trace_files_hp = [
    ("ResNet50", "rnet"),
    ("MobileNetV2", "mnet"),
]

trace_files_be = [
    ("ResNet50", "rnet"),
    ("MobileNetV2", "mnet"),
    ("ResNet101", "rnet101"),
    ("BERT", "bert"),
    ("Transformer", "trans")
]

for (model, f) in trace_files_hp:
    for run in range(num_runs):
        print(model, run, flush=True)
        # run
        file_path = f"config_files/ideal/{f}_inf.json"
        os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path}")

        # copy results
        os.system(f"cp client_0.json results/ideal/{model}_{run}_hp.json")
        os.system("rm client_0.json")

for (model, f) in trace_files_be:
    for run in range(num_runs):
        print(model, run, flush=True)
        # run
        file_path = f"config_files/ideal/{f}_train.json"
        os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path}")

        # copy results
        os.system(f"cp client_0.json results/ideal/{model}_{run}_be.json")
        os.system("rm client_0.json")
