import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
nsys_names = list(df['Name'])

nsys_kernel_names = [x for x in nsys_names if 'CUDA' not in x]
unique_kernel_names = set(nsys_kernel_names)

processed_kernel_names = []
rem_kernel_names = []

for i, row in df.iterrows():
    x = row['Name']
    if ('memset' in x) or ('memcpy' in x):
        continue
    #processed_kernel_names.append(x)

    if 'cudnn' in x:
        if 'bn_fw' in x:
            processed_kernel_names.append(['BatchNorm', row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])
        elif ('scudnn' in x) or ('implicit_convolve_sgemm' in x):
            processed_kernel_names.append(['Conv', row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])
        elif ('cudnn::winograd' in x) or ('cudnn::gemm' in x):
            # part of cudnn mm
            pass
        else:
            processed_kernel_names.append([x,  row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])

    elif 'volta_sgemm_128x64_nn' in x:
        processed_kernel_names.append(['Conv', row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])

    elif 'splitKreduce_kernel' in x:
        # part of cublas mm
        pass

    else:
        tokens = x.split('<')
        processed_kernel_names.append([tokens[0],  row['GrdX'], row['GrdY'], row['GrdZ'], row['BlkX'], row['BlkY'], row['BlkZ']])

for i,x in enumerate(processed_kernel_names):
    print(i,x)

with open('vgg16_bn', 'w') as f:
    for x in processed_kernel_names:
        f.write(x[0] + '\n')
