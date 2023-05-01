import os
import time

trace_files = [
    "rnet_rnet",
    "rnet_mnet",
    "rnet_rnet101",
    "rnet_bert",
    "rnet_trans",
    "mnet_rnet",
    "mnet_mnet",
    "mnet_rnet101",
    "mnet_bert",
    "mnet_trans",
    "rnet101_rnet",
    "rnet101_mnet",
    "rnet101_rnet101",
    "rnet101_bert",
    "rnet101_trans",
    "bert_rnet",
    "bert_mnet",
    "bert_rnet101",
    "bert_bert",
    "bert_trans",
    "trans_rnet",
    "trans_mnet",
    "trans_rnet101",
    "trans_bert",
    "trans_trans"
]


depths = [160000, 150000, 200000, 1250000, 400000,
          160000, 150000, 200000, 1250000, 400000,
          160000, 150000, 200000, 1250000, 400000,
          160000, 150000, 200000, 1250000, 400000,
          160000, 150000, 200000, 1250000, 400000]


# depths = [
#     6,5,10,48,16,
#     8,6,13,58,21,
#     8,6,13,60,21,
#     16,13,27,123,43,
#     22,17,36,170,59,
# ]

print(len(trace_files), len(depths))
assert len(trace_files) == len(depths)
for f,d in zip(trace_files, depths):
    print(f,d, flush=True)
    file_path = f"eval/train_inf/poisson/{f}_ti.json"
    os.system(f"python launch_jobs.py {file_path} {d}")
    time.sleep(10)
