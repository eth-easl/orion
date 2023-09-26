import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
args = parser.parse_args()

df = pd.read_csv(f'{args.results_dir}/output_ncu.csv', index_col=0)
kernels = []
metrics_to_get = ['Duration', 'Block Size', 'Grid Size', 'Compute (SM) [%]', 'DRAM Throughput', 'Registers Per Thread', 'Static Shared Memory Per Block']

unique_kernel_names = set()

for index, row in df.iterrows():
    kernel = row['Kernel Name']
    metric_name = row['Metric Name']

    if metric_name == 'DRAM Frequency':
        kernels.append([kernel])
        unique_kernel_names.add(kernel)
    elif metric_name in metrics_to_get:
        kernels[-1].append(row['Metric Value'])

for x in unique_kernel_names:
    print(x)
    print("------------------------------------")


for kernel in kernels:
    num_threads = int(kernel[-2]) * int(kernel[-3])
    num_registers = num_threads * int(kernel[-1])
    kernel += [num_threads, num_registers]


print(len(kernels))
#print(kernels[0])
labels = ['Kernel_Name', 'DRAM_Throughput(%)', 'Duration(ns)', 'Compute(SM)(%)',  'Block', 'Grid', 'Registers_Per_Thread', 'Static_shmem_per_block', 'Number_of_threads', 'Number_of_registers']



df_new = pd.DataFrame(kernels, columns=labels)
print(df_new)
df_new.to_csv(f'{args.results_dir}/output_ncu_processed.csv')
