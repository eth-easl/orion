import os
import time

num_runs = 3
trace_files = [
    ("ResNet50", "ResNet50", "rnet_rnet_ti"),
    ("ResNet50", "MobileNetV2", "rnet_mnet_ti"),
    ("MobileNetV2", "ResNet50", "mnet_rnet_ti"),
    ("MobileNetV2", "MobileNetV2", "mnet_mnet_ti"),
    ("ResNet101", "ResNet50", "rnet101_rnet_ti"),
    ("ResNet101", "MobileNetV2", "rnet101_mnet_ti"),
    ("BERT", "ResNet50", "bert_rnet_ti"),
    ("BERT", "MobileNetV2", "bert_mnet_ti"),
    ("Transformer", "ResNet50", "trans_rnet_ti"),
    ("Transformer", "MobileNetV2", "trans_mnet_ti"),
]

for (be, hp, f) in trace_files:

    for run in range(num_runs):

        # run
        file_path = f"config_files/{f}.json"
        os.system(f"LD_PRELOAD='/home/image-varuna/orion/src/cuda_capture/libinttemp.so' python ../../benchmarking/launch_jobs.py --algo sequential --config_file {file_path}")

        # copy results
        os.system(f"cp hp.json results/sequential/{be}_{hp}_{run}_hp.json")
        os.system(f"cp hp.json results/sequential/{be}_{hp}_{run}_be.json")