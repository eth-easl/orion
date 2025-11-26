import os
import time

num_runs = 1
trace_files = [
    ("ResNet50", "ResNet50", "rnet_rnet", 160000),
    ("MobileNetV2", "ResNet50", "mnet_rnet", 160000),
    ("ResNet101", "ResNet50", "rnet101_rnet", 160000),
    ("BERT", "ResNet50", "bert_rnet", 160000),

    ("ResNet50", "MobileNetV2", "rnet_mnet", 100000),
    ("MobileNetV2", "MobileNetV2", "mnet_mnet", 100000),
    ("ResNet101", "MobileNetV2", "rnet101_mnet", 100000),
    ("BERT", "MobileNetV2", "bert_mnet", 100000),

    ("ResNet50", "ResNet101", "rnet_rnet101", 320000),
    ("MobileNetV2", "ResNet101", "mnet_rnet101", 320000),
    ("ResNet101", "ResNet101", "rnet101_rnet101", 320000),
    ("BERT", "ResNet101", "bert_rnet101", 320000),

    ("ResNet50", "BERT", "rnet_bert", 2000000),
    ("MobileNetV2", "BERT", "mnet_bert", 2000000),
    ("ResNet101", "BERT", "rnet101_bert", 2000000),
    ("BERT", "BERT", "bert_bert", 2000000),
]

ld_preload_cmd = "LD_PRELOAD=/root/orion/src/cuda_capture/libinttemp.so:/usr/local/lib/python3.10/dist-packages/torch/lib/../../nvidia/cudnn/lib/libcudnn.so.9:/usr/local/lib/python3.10/dist-packages/torch/lib/../../nvidia/cublas/lib/libcublasLt.so.12:/usr/local/lib/python3.10/dist-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.12"

for (be, hp, f, max_be_duration) in trace_files:
    for run in range(num_runs):
        print(be, hp, run, flush=True)
        # run
        file_path = f"config_files/{f}.json"
        os.system(f"{ld_preload_cmd} python3 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration}")

         # copy results
        os.system(f"cp client_0.json results/orion/{be}_{hp}_{run}_be.json")
        os.system("rm client_0.json")

        # copy results
        os.system(f"cp client_1.json results/orion/{be}_{hp}_{run}_hp.json")
        os.system("rm -rf client_1.json")
