import pandas as pd
import sys
import os

pwd = sys.argv[1]
df = pd.read_csv(f'{pwd}/output_ncu_sms_roofline_fwd_only.csv')

kernels = []
needed = []
for index, row in df.iterrows():
    kernels.append([
        row['Kernel_Name'],
        row['Roofline_prof'],
        0,
        row['SM_needed'],
        row['Duration(ns)']
    ])
    needed.append(int(row['SM_needed']))


labels = ["Name", "Profile", "Memory_footprint", "SM_needed", "Duration"]
df_new = pd.DataFrame(kernels, columns=labels)
file_name = "resnet50_32_kernel_info.csv"
df_new.to_csv(f"{pwd}/{file_name}")

smaller = [x for x in needed if x < 80]
print(len(smaller), len(needed))

cmd = f"gcloud compute scp {pwd}/{file_name} image-varuna@vm-torch-3:/home/image-varuna/gpu_share_repo/scripts/ --project ml-elasticity"
print(cmd)
os.system(cmd)
