import pandas as pd
from math import ceil, floor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--max_threads_sm', type=int, default=2048,
                        help='maximum number of threads that can be active in an SM')
parser.add_argument('--max_blocks_sm', type=int, default=80,
                        help='maximum number of blocks that can be active in an SM')
parser.add_argument('--max_shmem_sm', type=int, default=65536,
                        help='maximum amount of shared memory (in bytes) per SM')
parser.add_argument('--max_regs_sm', type=int, default=65536,
                        help='maximum number of registers per SM')
args = parser.parse_args()

df = pd.read_csv(f'{args.results_dir}/output_ncu_processed.csv', index_col=0)

max_threads_sm = args.max_threads_sm
max_blocks_sm = args.max_blocks_sm
max_shmem_sm = args.max_shmem_sm
max_regs_sm = args.max_regs_sm

sm_needed = []

for index, row in df.iterrows():
    num_blocks = row['Grid']
    num_threads = row['Number_of_threads']
    threads_per_block = row['Block']
    shmem_per_block = row['Static_shmem_per_block']
    regs_per_thread = row['Registers_Per_Thread']

    # from threads
    blocks_per_sm_threads = ceil(max_threads_sm/threads_per_block)

    # from shmem
    if shmem_per_block > 0:
        blocks_per_sm_shmem = ceil(max_shmem_sm/shmem_per_block)
    else:
        blocks_per_sm_shmem = blocks_per_sm_threads

    # from registers
    regs_per_wrap = ceil(32*regs_per_thread/256) * 256
    wraps_per_sm = floor((65536/regs_per_wrap)/4) * 4
    wraps_per_block = ceil(threads_per_block/32)
    blocks_per_sm_regs = int(wraps_per_sm/wraps_per_block)

    blocks_per_sm = min(blocks_per_sm_threads, blocks_per_sm_shmem, blocks_per_sm_regs)
    sm_needed_kernel = ceil(num_blocks/blocks_per_sm)

    #print(blocks_per_sm, sm_needed_kernel)
    sm_needed.append(sm_needed_kernel)


less = [x for x in  sm_needed if x < 108]
print(len(less), len(sm_needed))

df['SM_needed'] = sm_needed
#print(df)
df.to_csv(f'{args.results_dir}/output_ncu_sms.csv', index=0)
