import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import sys

def get_times(start_times, dur, sm_used):

    new_times = []
    new_sm = []

    times_sm_all = []
    sz = len(start_times)

    for i in range(sz):
        sti = start_times[i]
        di = dur[i] * 1e-9
        smi = sm_used[i]
        #print(smi)
        times_sm_all.append([sti, smi])
        times_sm_all.append([sti+di, -smi])

    times_sm_all = sorted(times_sm_all)
    print(times_sm_all)

    cur = 0
    for x in times_sm_all:
        cur += x[1]
        #print(cur)
        new_times.append(x[0])
        new_sm.append(min(cur, 80))


    return new_times, new_sm


plot_mem = False
plot_grid_size = False
plot_compute = False
plot_comp_mem = False
plot_sms = True

pwd =sys.argv[1]
df_nsys = pd.read_csv(f'{pwd}/output_nsys_gputrace.csv')
#df_nsys = pd.read_csv(f'ResNet50-ImageNet-BS16/nsight_report_16_gputrace_profile.csv')
df_ncu = pd.read_csv(f'{pwd}/output_ncu_sms_roofline.csv')

start_time_all = df_nsys['Start(sec)']
dur_all = df_nsys['Duration(nsec)']
# grdx = df_nsys['GrdX']
# grdy = df_nsys['GrdY']
# grdz = df_nsys['GrdZ']
# blockx = df_nsys['BlkX']
# blocky = df_nsys['BlkY']
# blockz = df_nsys['BlkZ']
# names = df_nsys.iloc[:,21]


mem = list(df_ncu['DRAM_Throughput(%)'])
comp = list(df_ncu['Compute(SM)(%)'])
sms_needed_all = list(df_ncu['SM_needed'])

names = df_nsys.iloc[:,19]
print(names)

grid_size = []
block_size = []
start_time = []
sm_used = []
sm_needed = []

names_new = []
#grid_ncu = list(df_ncu['Grid'])
dur = []
i=0

for j in range(len(names)):
    if 'memcpy' not in names[j] and 'memset' not in names[j]:

        #print(num_threads, blocks_per_sm, sm_needed_threads, sm_needed_blocks, sms)
        sms = sms_needed_all[i]
        print(min(sms, 80))
        sm_used.append(min(sms, 80))
        sm_needed.append(sms)
        #grid_size.append(num_blocks)
        #block_size.append(num_threads)
        start_time.append(start_time_all[i])
        dur.append(dur_all[i])
        names_new.append(names[i])
        i += 1



df_new = pd.DataFrame()
df_new['Start(sec)'] = start_time
df_new['Duration(nsec)'] = dur
# df_new['Blocks'] = grid_size
# df_new['Threads_per_Block'] = block_size
df_new['SMs'] = sm_used
df_new['SMs needed'] = sm_needed
df_new['Names'] = names_new

print(df_new)


#df_new.to_csv(f'{pwd}/nsight_report_16_sequential_selected.csv', index=0)


#print(grid_size[25:27], grid_ncu[25:27])
#assert grid_size[:27] == grid_ncu[:27]
#print(len(grid_size), len(df_ncu['Grid']))
#assert grid_size == list(df_ncu['Grid'])


### grid_size

base_title = 'ResNet50, Infer, BS 4'

if plot_mem:

    fig, ax1 = plt.subplots(figsize=(20,6))

    #start_time_next = start_time[400:]
    #mem = mem[400:]
    #start_time_next.append(start_time[-1] + dur[-1]*10e-9)
    #start_time = start_time_next

    start_time.append(start_time[-1] + dur[-1]*1e-9)

    current = 0
    all = 0
    for i in range(len(start_time)-1):
        current += mem[i] * (start_time[i+1]-start_time[i])
        all += 100 * (start_time[i+1]-start_time[i])
    avg = current*100/all
    print("current is: ", current)
    print("overall is: ", all)

    print(f"average MEM util is {avg} %")

    plt.stairs(mem, start_time) #, marker="o")
    plt.hlines(y=avg, xmin=start_time[0], xmax=start_time[-1], colors='red', linestyles='dashed', label='average')

    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('Memory bandwidth usage (%)', fontsize=22)

    #print(mem[-10:], start_time[-10:])

    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.title(base_title, fontsize=18)
    plt.savefig(f'{pwd}/mem_over_time_2.svg', bbox_inches="tight")


elif plot_compute:


    #start_time_next = start_time[:200]
    #comp = comp[:200]
    #start_time_next.append(start_time[199] + dur[199]*10e-9)
    #start_time = start_time_next

    fig, ax1 = plt.subplots(figsize=(20,6))

    start_time.append(start_time[-1] + dur[-1]*1e-9)

    #total = 80 * (start_time[-1] - start_time[0])
    current = 0
    all = 0
    for i in range(len(start_time)-1):
        current += comp[i] * (start_time[i+1]-start_time[i])
        all += 100 * (start_time[i+1]-start_time[i])
    avg = current*100/all
    print("current is: ", current)
    print("overall is: ", all)

    print(f"average COMP util is {avg} %")

    plt.stairs(comp, start_time) #, marker="o")
    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('Compute throughput (%)', fontsize=22)

    plt.hlines(y=avg, xmin=start_time[0], xmax=start_time[-1], colors='red', linestyles='dashed', label='average')

    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #print(mem[-10:], start_time[-10:])

    sumi = 0
    total = 0
    for i in range(len(comp)):
        sumi += comp[i] * dur[i]
        total += dur[i]

    avg = sumi/total
    print(avg)
    #plt.title(base_title, fontsize=22)


    plt.savefig(f'{pwd}/comp_over_time_2.svg', bbox_inches="tight")


elif plot_comp_mem:

     fig, ax1 = plt.subplots(figsize=(20,8))

     start_time = start_time[-150:]
     mem = mem[-150:]
     comp = comp[-150:]

     ax1.step(start_time, comp) #marker="o")
     ax1.set_xlabel('Time (sec)', fontsize=18)
     ax1.set_ylabel('Compute Troughput (%)', fontsize=18)

     ax2=ax1.twinx()
     ax2.step(start_time, mem, color='red')
     ax2.set_ylabel('Memory bw usage (%)', fontsize=18)

     #plt.title(base_title)
     plt.savefig(f'{pwd}/comp_mem_over_time_last150.png', bbox_inches="tight")



elif plot_grid_size:

    smaller = [x for x in grid_size if x < 2560]

    sp = len(smaller)*100/len(grid_size)
    print(f"smaller perc: {sp}")

    fig, ax1 = plt.subplots(figsize=(12,8))
    plt.plot(start_time, grid_size, marker="o")
    ax1.set_xlabel('Time (sec)', fontsize=20)
    ax1.set_ylabel('Number of Blocks')

    plt.title(base_title + ' sp={:.2f}'.format(sp))
    plt.savefig(f'{pwd}/blocks_over_time.png', bbox_inches="tight")


elif plot_sms:

    smaller = [x for x in sm_used if x < 80]

    times, sm_used = get_times(start_time, dur, sm_used)
    print(times, sm_used)

    sm_perc = len(smaller)*100/len(sm_needed)
    print(f"smaller perc: {sm_perc}")

    #times = start_time
    all = 0
    current = 0
    for i in range(len(times)-1):
        current += sm_used[i] * (times[i+1]-times[i])
        all += 80 * (times[i+1]-times[i])
    avg = (current-0.1)*100/all

    print("current is: ", current)
    print("overall is: ", all)
    print(f"average SM util is {avg} %")

    fig, ax1 = plt.subplots(figsize=(20,6))
    #plt.plot(times, sm_used, marker="o")

    ax1.step(times, sm_used)

    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('Number of SMs used', fontsize=22)

    #plt.hlines(y=80, xmin=times[0], xmax=times[-1], color='green', linestyle='--', label='SMs in V100')
    plt.hlines(y=avg*80/100, xmin=times[0], xmax=times[-1], color='red', linestyle='--', label='average')

    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper right')
    plt.title(base_title,fontsize=18)
    plt.savefig(f'{pwd}/sm_over_time_used.svg', bbox_inches="tight")
