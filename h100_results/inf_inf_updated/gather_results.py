import pandas as pd
import numpy as np
import json
import itertools
import os

models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']
baselines = ['temporal', 'orion' , 'reef', 'ideal']

hp_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']
be_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']
num_runs = 1

df_lat = pd.DataFrame("0", index=models, columns=models)
df_thr = pd.DataFrame("0", index=models, columns=models)

df_hp_ideal_throughput = pd.DataFrame("0", index=models, columns=models)
df_be_ideal_throughput = pd.DataFrame("0", index=models, columns=models)
for hp in hp_list:
    res_hp = []
    res_lat = []
    for run in range(num_runs):
        input_file_hp = f"results/ideal/{hp}_{run}_hp.json"
        with open(input_file_hp, 'r') as f:
            data = json.load(f)
            res_hp.append(float(data['throughput']))
            res_lat.append(float(data['p95_latency']))
    for be in be_list:
        df_hp_ideal_throughput.at[be, hp] = f"{round(np.average(res_hp),2)}/{round(np.std(res_hp),2)}"
        df_lat.at[be, hp] = f"{round(np.average(res_lat),2)}/{round(np.std(res_lat),2)}"

for be in be_list:
    res_be = []
    for run in range(num_runs):
        input_file_be = f"results/ideal/{be}_{run}_hp.json"
        with open(input_file_be, 'r') as f:
            data = json.load(f)
            res_be.append(float(data['throughput']))
    for hp in hp_list:
        df_be_ideal_throughput.at[be, hp] = f"{round(np.average(res_be),2)}/{round(np.std(res_be),2)}"

df_hp_ideal_throughput.to_csv(f'results/ideal_hp_throughput.csv')
df_be_ideal_throughput.to_csv(f'results/ideal_be_throughput.csv')
df_lat.to_csv(f'results/ideal_latency.csv')

print("ideal")
print(df_hp_ideal_throughput)
print(df_be_ideal_throughput)

# mps
df_mps = pd.DataFrame(0.0, index=models, columns=models)
for hp in hp_list:
    for be,hp in itertools.product(be_list, hp_list):
        results = []
        for run in range(num_runs):
            input_file = f"results/mps/{hp}_{be}_{run}.json"
            with open(input_file, 'r') as f:
                data = json.load(f)
                if data:
                    results.append(float(data['p95-latency-0']))
        df_mps.at[be, hp] = f"{round(np.average(results),2)}/{round(np.std(results),2)}"
df_mps.to_csv(f'results/mps_latency.csv')
print("------------- mps")
print(df_mps)

df_hp_mps_throughput = pd.DataFrame("0", index=models, columns=models)
df_be_mps_throughput = pd.DataFrame("0", index=models, columns=models)
for be,hp in itertools.product(be_list, hp_list):
    res_hp = []
    res_be = []
    for run in range(num_runs):
        input_file_hp = f"results/mps/{hp}_{be}_{run}.json"
        with open(input_file_hp, 'r') as f:
            data = json.load(f)
            if data:
                res_be.append(float(data['throughput-1']))
                res_hp.append(float(data['throughput-0']))

    df_hp_mps_throughput.at[be, hp] = f"{round(np.average(res_hp),2)}/{round(np.std(res_hp),2)}"
    df_be_mps_throughput.at[be, hp] = f"{round(np.average(res_be),2)}/{round(np.std(res_be),2)}"

df_hp_mps_throughput.to_csv(f'results/mps_hp_throughput.csv')
df_be_mps_throughput.to_csv(f'results/mps_be_throughput.csv')
print("mps")
print(df_hp_mps_throughput)
print(df_be_mps_throughput)

for baseline in baselines[:-1]:
    df_lat_hp = pd.DataFrame("0", index=models, columns=models)
    df_thr_hp = pd.DataFrame("0", index=models, columns=models)

    df_thr_be = pd.DataFrame("0", index=models, columns=models)

    for be,hp in itertools.product(be_list, hp_list):
        results_lat_hp = []
        results_thr_hp = []

        results_be = []
        for run in range(num_runs):
            input_file_hp = f"results/{baseline}/{be}_{hp}_{run}_hp.json"
            if not os.path.exists(input_file_hp):
                continue

            with open(input_file_hp, 'r') as f:
                data = json.load(f)
                results_lat_hp.append(float(data['p95_latency']))
                results_thr_hp.append(float(data['throughput']))

            input_file_be = f"results/{baseline}/{be}_{hp}_{run}_be.json"
            with open(input_file_be, 'r') as f:
                data = json.load(f)
                results_be.append(float(data['throughput']))

        df_lat_hp.at[be, hp] = f"{round(np.average(results_lat_hp),2)}/{round(np.std(results_lat_hp),2)}"
        df_thr_hp.at[be, hp] = f"{round(np.average(results_thr_hp),2)}/{round(np.std(results_thr_hp),2)}"
        df_thr_be.at[be, hp] = f"{round(np.average(results_be),2)}/{round(np.std(results_be),2)}"

    df_lat_hp.to_csv(f'results/{baseline}_latency.csv')
    df_thr_be.to_csv(f'results/{baseline}_be_throughput.csv')
    df_thr_hp.to_csv(f'results/{baseline}_hp_throughput.csv')
    print(f"----------- {baseline}")
    print(df_lat_hp)
    print(df_thr_be)