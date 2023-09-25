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

#orion, hp is inference - uniform
#depths = [160000, 150000, 200000, 1250000, 400000,
#          160000, 150000, 200000, 1250000, 400000,
#          160000, 150000, 200000, 1250000, 400000,
#          160000, 150000, 200000, 1250000, 400000,
#          160000, 150000, 200000, 1250000, 400000]


# reef depths
depths = [
     15,12,26,130,43,
     40,30,70,329,114,
     17,13,30,140,50,
     7,5,11,56,19,
     16,13,27,130,45,
]



# orion, hp is inference, threshold is 0.05
# depths = [320000, 300000, 400000, 2500000, 800000,
#           320000, 300000, 400000, 2500000, 800000,
#           320000, 300000, 400000, 2500000, 800000,
#           320000, 300000, 400000, 2500000, 800000,
#           320000, 300000, 400000, 2500000, 800000]

limits = [1,1,1,1,1,
          1,1,1,1,1,
          1,1,1,1,1,
          1,1,1,1,1,
          1,1,1,1,1]
updates = [1,1,1,1,1,
           1,1,1,1,1,
           1,1,1,1,1,
           1,1,1,1,1,
           1,1,1,1,1]


# # orion, hp is training
# depths = [
#     1000000, 1000000, 1000000, 40000000, 32000000,
#     1000000, 1000000, 1000000, 40000000, 32000000,
#     1000000, 1000000, 1000000, 40000000, 32000000,
#     1000000, 1000000, 1000000, 40000000, 32000000,
#     1000000, 1000000, 1000000, 40000000, 32000000
# ]
# limits = [
#     135, 120, 235, 250, 250,
#     135, 120, 235, 250, 250,
#     135, 120, 235, 250, 250,
#     135, 120, 235, 250, 250,
#     135, 120, 235, 250, 250
# ]
# updates = [
#     768, 733, 1534, 2669, 1622,
#     768, 733, 1534, 2669, 1622,
#     768, 733, 1534, 2669, 1622,
#     768, 733, 1534, 2669, 1622,
#     768, 733, 1534, 2669, 1622
# ]

# depths = [
#     6,5,10,48,16,
#     8,6,13,58,21,
#     8,6,13,60,21,
#     16,13,27,123,43,
#     22,17,36,170,59,
# ]

print(len(trace_files), len(depths))
assert len(trace_files) == len(depths)
for f,d,l,u in zip(trace_files, depths, limits, updates):
    print(f,d, flush=True)
    file_path = f"eval/train_train/{f}.json"
    os.system(f"python launch_jobs.py {file_path} {d} {l} {u}")
    time.sleep(10)
