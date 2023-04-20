import os
import time

trace_files = [
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
depths = [8,6,13,60,21,16,13,27,123,43,22,17,36,170,59]

assert len(trace_files) == len(depths)
for f,d in zip(trace_files, depths):
    print(f,d, flush=True)
    file_path = f"eval/train_inf/{f}_ti.json"
    os.system(f"python launch_jobs.py {file_path} {d}")
    time.sleep(10)
