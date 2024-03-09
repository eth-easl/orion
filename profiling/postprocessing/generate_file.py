import pandas as pd
import numpy as np
import argparse


def get_profile(profile_list, main_prof):
    pset = set(profile_list)
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


parser = argparse.ArgumentParser()
parser.add_argument('--input_file_name', type=str, required=True,
                        help='path to the profiling input file')
parser.add_argument('--output_file_name', type=str, required=True,
                        help='path to the profiling output file')
parser.add_argument('--model_type', type=str, required=True,
                        help='Model type. (vision, bert, transformer) are currently supported')
args = parser.parse_args()

df = pd.read_csv(args.input_file_name)
output_file_name = args.output_file_name

# nsys_names = list(df['Name'])

# nsys_kernel_names = [x for x in nsys_names if 'CUDA' not in x]
# unique_kernel_names = set(nsys_kernel_names)

processed_kernel_names = []
rem_kernel_names = []

conv_info = []
l = df.to_dict('records')
print(len(l))

found = 0
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
    print(x)

    if 'cudnn' in x and 'LSTM' not in x:
        print(x)
        if ('bn_fw' in x) or ('bn_bw' in x):
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
            or ('cutlass_cudnn::Kernel' in x)
            or ('xmma_cudnn::gemm::kernel' in x)
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
            or ('computeOffsetsKernel' in x)
            or ('scalePackedTensor_kernel') in x
            or ('fft' in x)
            or ('nchwToNhwcKernel' in x)
            or ('nhwcToNchwKernel' in x)
            or ('nhwcToFoldedNhwcKernel' in x)
            or ('foldedNhwcToNhwcKernel' in x)
            or ('nhwcAddPaddingKernel' in x)
            or ('im2col4d_kernel' in x)
        ):
            # part of cudnn mm
            conv_info.append([row["SM_needed"], row["Duration(ns)"], row["Roofline_prof"]])
        else:
            processed_kernel_names.append([x,  row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])

    elif ('sm80_xmma' in x or 'implicit_convolve_sgemm' in x):
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

    elif args.model_type == 'vision' and (('volta_sgemm_128x64_nn' in x) or ('volta_sgemm_128x64_nt' in x)):
          processed_kernel_names.append(['Conv', row['Roofline_prof'], 0, row["SM_needed"], row["Duration(ns)"]])

    elif 'volta_gcgemm_32x32_nt' in x:
        if found==0:
            processed_kernel_names.append(['Conv', row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])
        found = 1

    elif 'splitKreduce_kernel' in x:
        # part of cublas mm
        found = 0
        pass

    elif args.model_type == 'transformer' and ('volta_sgemm_32x128_tn' in x or 'ampere_sgemm_32x128_tn'):
        # check next row
        if i < num_rows-1:
            next_row = l[i+1]
            sms = row["SM_needed"]
            duration = row["Duration(ns)"]
            profile = row["Roofline_prof"]
            if 'splitKreduce_kernel' in next_row['Kernel_Name']:
                sms = max(sms, next_row["SM_needed"])
                duration += next_row["Duration(ns)"]
                profile = get_profile([profile, next_row["Roofline_prof"]], profile)
        processed_kernel_names.append([x, profile, 0, sms, duration])


    elif args.model_type == 'transformer' and 'volta_gcgemm_32x32_nt' in x:
        # vision
        if found==3:
            print("here!")
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
        found += 1

    else:
        tokens = x.split('<')
        found = 0
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
