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
cublas_list = [26, 46]
for i, row in df.iterrows():
    x = row['Name']
    if ('memset' in x) or ('memcpy' in x):
        continue

    if ('splitKreduce_kernel' in x) or ('gemv2T_kernel_val' in x):
        continue

    if ('LSTM' not in x) and ('sgemm' not in x) and ('transpose_readWrite_alignment_kernel') not in x:
        tokens = x.split('<')
        processed_kernel_names.append([tokens[0],  row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])
        idx += 1

        if idx in cudnn_list:
            processed_kernel_names.append("cudnn_rnn")
        elif idx == 26:
            processed_kernel_names.append("cublas_sgemm")
            processed_kernel_names.append("cublas_sgemm")
        elif idx==46:
            processed_kernel_names.append("cublas_sgemm")

'''
for i, row in df.iterrows():
    x = row['Name']
    if ('memset' in x) or ('memcpy' in x):
        continue
    #processed_kernel_names.append(x)

    if 'cudnn' in x and 'LSTM' not in x:
        if 'bn_fw' in x:
            processed_kernel_names.append(['BatchNorm', row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        elif ('scudnn' in x) or ('implicit_convolve_sgemm' in x):
            processed_kernel_names.append(['Conv', row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])
        elif ('cudnn::winograd' in x) or ('cudnn::gemm' in x):
            # part of cudnn mm
            pass
        else:
            processed_kernel_names.append([x,  row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])

    elif 'volta_sgemm_128x64_nn' in x:
        processed_kernel_names.append(['Conv', row['Profile'], row["Memory_footprint"], row["SM_needed"], row["Duration"]])

    elif 'splitKreduce_kernel' in x:
        # part of cublas mm
        pass

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
    for x in processed_kernel_names:
        f.write(x[0] + '\n')
