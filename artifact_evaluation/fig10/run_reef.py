import os
import time

num_runs = 3
trace_files = [
    ("ResNet50", "ResNet50", "rnet_rnet"),
    ("ResNet50", "MobileNetV2", "rnet_mnet"),
    ("MobileNetV2", "ResNet50", "mnet_rnet"),
    ("MobileNetV2", "MobileNetV2", "mnet_mnet"),
    ("ResNet101", "ResNet50", "rnet101_rnet"),
    ("ResNet101", "MobileNetV2", "rnet101_mnet"),
    ("BERT", "ResNet50", "bert_rnet"),
    ("BERT", "MobileNetV2", "bert_mnet"),
    ("Transformer", "ResNet50", "trans_rnet"),
    ("Transformer", "MobileNetV2", "trans_mnet"),
]

for (be, hp, f) in trace_files:
    for run in range(num_runs):
        print(be, hp, run, flush=True)
        # run
        file_path = f"config_files/{f}.json"
        os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.8 ../../benchmarking/launch_jobs.py --algo reef --config_file {file_path} --reef_depth 12")

        # copy results
        os.system(f"cp client_1.json results/reef/{be}_{hp}_{run}_hp.json")
        os.system("rm client_1.json")
