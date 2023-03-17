import pandas as pd
from math import ceil, floor
import sys

pwd = sys.argv[1]
df = pd.read_csv(f'{pwd}/output_ncu_processed.csv', index_col=0)

# for V100
max_threads_sm = 2048
max_blocks_sm = 80
max_shmem_sm = 65536
max_regs_sm = 65536

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


less = [x for x in  sm_needed if x < 80]
print(len(less), len(sm_needed))

df['SM_needed'] = sm_needed
#print(df)
df.to_csv(f'{pwd}/output_ncu_sms.csv', index=0)
