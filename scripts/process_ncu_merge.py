import pandas as pd
import sys
import numpy as np

def get_profile(profile_list, main_prof):
    pset = set(profile_list)
    print(pset)
    if -1 in pset:
        pset.remove(-1)
    if len(pset)==0:
        return -1
    if pset == {0}:
        return 0
    if pset == {1}:
        return 1
    if pset == {0,1}:
        return main_prof


df = pd.read_csv(sys.argv[1])
output_file_name = sys.argv[2]

# nsys_names = list(df['Name'])

# nsys_kernel_names = [x for x in nsys_names if 'CUDA' not in x]
# unique_kernel_names = set(nsys_kernel_names)

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
l = df.to_dict('records')
print(len(l))

i = 0
num_rows = len(l)
while i < num_rows:
    row = l[i]
    x = row['Kernel_Name']
    if ('memset' in x) or ('memcpy' in x):
        i += 1
        continue
    #processed_kernel_names.append(x)

    x = x.replace("<unnamed>", "(anonymous namespace)")

    if 'cudnn' in x and 'LSTM' not in x:
        if 'bn_fw' in x:
            processed_kernel_names.append(['BatchNorm', row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])
        elif (
            ('scudnn' in x)
            or ('implicit_convolve_sgemm' in x)
            or ('explicit_convolve_sgemm' in x)
            or ('dgrad_engine' in x)
            or ('wgrad_alg0_engine' in x)
            or ('wgrad_alg1_engine_NHWC' in x)
            or ('dgrad2d_alg1_1' in x)
            or ('wgrad2d_grouped_direct_kernel' in x)
            or ('dgrad2d_grouped_direct_kernel' in x)
            or ('conv2d_grouped_direct_kernel' in x)
            or ('convolve_common_engine_float_NHWC' in x)
        ):
            conv_info.append([row["SM_needed"], row["Duration(ns)"], row["Roofline_prof"]])
            print(conv_info)
            sms = [x[0] for x in conv_info]
            dur_list = [x[1] for x in conv_info]
            profiles = [x[2] for x in conv_info]
            sms_max = max(sms)
            dur = sum(dur_list)
            profile = get_profile(profiles,  row["Roofline_prof"])
            processed_kernel_names.append(['Conv', profile, 0, sms_max, dur])
            conv_info=[]
        elif (
            ('cudnn::winograd' in x)
            or ('cudnn::gemm' in x)
            or ('scalePackedTensor_kernel') in x
            or ('fft' in x)
            or ('nhwcToFoldedNhwcKernel' in x)
            or ('foldedNhwcToNhwcKernel' in x)
            or ('nhwcAddPaddingKernel' in x)
            or ('im2col4d_kernel' in x)
        ):
            # part of cudnn mm
            conv_info.append([row["SM_needed"], row["Duration(ns)"], row["Roofline_prof"]])
        else:
            processed_kernel_names.append([x,  row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])

    #Comment for NLP models
    # elif ('volta_sgemm_128x64_nn' in x) or ('volta_sgemm_128x64_nt' in x):
    #     processed_kernel_names.append(['Conv', row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])

    # transformer
    elif 'volta_sgemm_32x128_tn' in x:
        # check next row
        next_row = l[i+1]
        sms = row["SM_needed"]
        duration = row["Duration(ns)"]
        profile = row["Roofline_prof"]
        # if 'splitKreduce_kernel' in next_row['Kernel_Name']:
        #     sms = max(sms, next_row["SM_needed"])
        #     duration += next_row["Duration(ns)"]
        #     profile = get_profile([profile, next_row["Roofline_prof"]], profile)
        processed_kernel_names.append([x, profile, 0, sms, duration])

    elif 'splitKreduce_kernel' in x:
        # part of cublas mm
        pass
    elif 'fused_dropout_kernel_vec' in x:
        processed_kernel_names.append(['fused_dropout_kernel_vec', row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])

    else:
        tokens = x.split('<')
        #print(tokens[0])
        processed_kernel_names.append([tokens[0],  row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])
    i += 1

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
