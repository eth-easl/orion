import pandas as pd
import sys
import numpy as np

df = pd.read_csv(sys.argv[1])
output_file_name = sys.argv[2]

nsys_names = list(df['Name'])

nsys_kernel_names = [x for x in nsys_names if 'CUDA' not in x]
unique_kernel_names = set(nsys_kernel_names)

processed_kernel_names = []
rem_kernel_names = []

# this is for gnmt, since it is a bit strange
'''
idx = 0
cudnn_list = [11, 14, 16, 19, 25, 37, 40, 44]
cublas_list = [41]

for i, row in df.iterrows():
    x = row['Name']
    
    x = x.replace("<unnamed>", "(anonymous namespace)")
    
    if ('memset' in x) or ('memcpy' in x):
        continue

    if ('splitKreduce_kernel' in x) or ('gemv2T_kernel_val' in x):
        continue

    if ('LSTM' not in x) and ('sgemm' not in x) and ('transpose_readWrite_alignment_kernel') not in x:
        tokens = x.split('<')
        processed_kernel_names.append([tokens[0],  row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        idx += 1

        if idx in cudnn_list:
            processed_kernel_names.append(["cudnn_rnn", row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        elif idx == 26:
            print(i)
            processed_kernel_names.append(["cublas_sgemm", row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
            processed_kernel_names.append(["cublas_sgemm", row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        elif idx==46:
            processed_kernel_names.append(["cublas_sgemm", row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
    elif 'sgemm' in x:
        if idx == 34:
            processed_kernel_names.append(["cublas_sgemm", row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])

'''
conv_info = []

for i, row in df.iterrows():
    x = row['Name']
    if ('memset' in x) or ('memcpy' in x):
        continue
    #processed_kernel_names.append(x)

    x = x.replace("<unnamed>", "(anonymous namespace)")

    if 'cudnn' in x and 'LSTM' not in x:
        if 'bn_fw' in x:
            processed_kernel_names.append(['BatchNorm', row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        elif ('scudnn' in x) or ('implicit_convolve_sgemm' in x) or ('explicit_convolve_sgemm' in x):
            conv_info.append([row["SM_needed"], row["Duration"]])
            dur_list = [x[1] for x in conv_info]
            sms = [x[0] for x in conv_info]
            sms_max = max(sms)
            dur = sum(dur_list)
            processed_kernel_names.append(['Conv', row['Profile'], row["Memory_footprint"], sms_max, dur])
            conv_info=[]
        elif ('cudnn::winograd' in x) or ('cudnn::gemm' in x):
            # part of cudnn mm
            conv_info.append([row["SM_needed"], row["Duration"]])
        elif ('im2col4d_kernel' in x):
            # part of conv
            pass
        else:
            processed_kernel_names.append([x,  row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])

    elif 'volta_sgemm_128x64_nn' in x:
        processed_kernel_names.append(['Conv', row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])

    elif 'splitKreduce_kernel' in x:
        # part of cublas mm
        pass
    elif 'fused_dropout_kernel_vec' in x:
        processed_kernel_names.append(['fused_dropout_kernel_vec', row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        
    else:
        tokens = x.split('<')
        #print(tokens[0])
        processed_kernel_names.append([tokens[0],  row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])

sms_needed = []
for i,x in enumerate(processed_kernel_names):
    print(i,x)
    sms_needed.append(x[3])

smaller = [x for x in sms_needed if x < 80]
print(len(smaller), len(sms_needed))

with open(output_file_name, 'w') as f:
    f.write("Name,Profile,Memory_footprint,SM_usage,Duration\n");
    for x in processed_kernel_names:
        str_x = ",".join(str(y) for y in x)
        f.write(str_x+"\n")
