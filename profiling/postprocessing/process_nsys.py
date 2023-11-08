import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--max_sms', type=int, default=80,
                        help='Number of SMs in the GPU')
parser.add_argument('--metric', type=str, default="SM",
                        help='Which metric to plot: Could be either of "SM", "Comp", "Mem"')
args = parser.parse_args()

def get_times(start_times, dur, sms, threshold):

    new_times = []
    new_sm = []

    times_sm_all = []
    sz = len(start_times)

    for i in range(sz):
        sti = start_times[i]
        di = dur[i] * 1e-9
        smi = sms[i]
        times_sm_all.append([sti, smi])
        times_sm_all.append([sti+di, -smi])

    times_sm_all = sorted(times_sm_all)

    cur = 0
    for x in times_sm_all:
        cur += x[1]
        #print(cur)
        new_times.append(x[0])
        new_sm.append(cur)

    total = len(new_sm)
    for i in range(total):
        new_sm[i] = min(new_sm[i], threshold)

    return new_times, new_sm

plot_mem = args.metric == "Mem"
plot_compute = args.metric == "Comp"
plot_sms = args.metric == "SM"
max_sms = args.max_sms

pwd = args.results_dir
df_nsys = pd.read_csv(f'{pwd}/output_nsys_gputrace.csv')
df_ncu = pd.read_csv(f'{pwd}/output_ncu_sms_roofline.csv')

start_time_all = df_nsys['Start(sec)']
dur_all = df_nsys['Duration(nsec)']
grdx = df_nsys['GrdX']
grdy = df_nsys['GrdY']
grdz = df_nsys['GrdZ']
blockx = df_nsys['BlkX']
blocky = df_nsys['BlkY']
blockz = df_nsys['BlkZ']

mem = list(df_ncu['DRAM_Throughput(%)'])
comp = list(df_ncu['Compute(SM)(%)'])
sms_needed_all = list(df_ncu['SM_needed'])
ncu_names = list(df_ncu['Kernel_Name'])
ncu_duration = list(df_ncu['Duration(ns)'])
blocks = list(df_ncu['Block'])
grids = list(df_ncu['Grid'])
names_ncu = list(df_ncu['Kernel_Name'])

names = df_nsys.iloc[:,19]
print(names)

grid_size = []
block_size = []
start_time = []
sm_used = []
sm_needed = []

names_new = []
dur = []

i=0

mem_new = []
comp_new = []

for j in range(len(names)):
    if 'memcpy' not in names[j] and 'memset' not in names[j]:
        kname_nsys = names[j].split("<")[0]
        names_ncu[i] = names_ncu[i].replace('<unnamed>', '(anonymous namespace)')
        kname_ncu = names_ncu[i].split("<")[0]
        if kname_nsys != kname_ncu:
            print(i, j, kname_nsys, kname_ncu)

        sms = sms_needed_all[i]
        sm_used.append(min(sms, max_sms))
        sm_needed.append(sms)
        start_time.append(start_time_all[j])
        dur.append(dur_all[j])
        names_new.append(names[j])
        mem_new.append(mem[i])
        comp_new.append(comp[i])
        i += 1
        if i == len(ncu_names):
             break

print(len(sm_used))

mem = mem_new
comp = comp_new

df_new = df_ncu
df_new['Start(sec)'] = start_time
df_new['Duration(ns)'] = dur
df_new.to_csv(f'{pwd}/output_ncu_nsys.csv')

if plot_mem:

    fig, ax1 = plt.subplots(figsize=(20,6))

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

    print(len(start_time), len(mem))

    plt.stairs(mem, start_time, linewidth=1.5) #, marker="o")
    plt.hlines(y=avg, xmin=start_time[0], xmax=start_time[-1], colors='red', linestyles='dashed', label='average', linewidth=3.5)

    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('Memory usage (%)', fontsize=22)
    ax1.set_ylim(0,102)

    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'{pwd}/mem_over_time.png', bbox_inches="tight")


elif plot_compute:

    fig, ax1 = plt.subplots(figsize=(20,6))

    start_time.append(start_time[-1] + dur[-1]*1e-9)

    current = 0
    all = 0
    for i in range(len(start_time)-1):
        current += comp[i] * (start_time[i+1]-start_time[i])
        all += 100 * (start_time[i+1]-start_time[i])
    avg = current*100/all
    print("current is: ", current)
    print("overall is: ", all)

    print(f"average COMP util is {avg} %")

    plt.stairs(comp, start_time, linewidth=1.5) #, marker="o")
    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('Compute throughput (%)', fontsize=22)
    ax1.set_ylim(0,102)

    plt.hlines(y=avg, xmin=start_time[0], xmax=start_time[-1], colors='red', linestyles='dashed', label='average', linewidth=3.5)

    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(f'{pwd}/comp_over_time.png', bbox_inches="tight")

elif plot_sms:

    smaller = [x for x in sm_needed if x < max_sms]
    times, sm_used = get_times(start_time, dur, sm_needed, max_sms)
    sm_perc = len(smaller)*100/len(sm_needed)

    all = 0
    current = 0
    for i in range(len(times)-1):
        current += sm_used[i] * (times[i+1]-times[i])
        all += max_sms * (times[i+1]-times[i])
    avg = (current)*100/all

    print("current is: ", current)
    print("overall is: ", all)
    print(f"average SM util is {avg} %")

    fig, ax1 = plt.subplots(figsize=(20,6))
    sm_used_perc = [x*100/max_sms for x in sm_used]

    ax1.step(times, sm_used_perc)

    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('% SMs busy', fontsize=22)

    plt.hlines(y=avg, xmin=times[0], xmax=times[-1], color='red', linestyle='--', label='average')

    plt.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper right')
    plt.savefig(f'{pwd}/sm_over_time.png', bbox_inches="tight")
