import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']
baselines = ['sequential', 'streams', 'mps', 'reef', 'orion', 'ideal']

hp_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']
be_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']
num_runs = 1

df_hp_ideal_throughput = pd.DataFrame(0.0, index=models, columns=models)
df_be_ideal_throughput = pd.DataFrame(0.0, index=models, columns=models)
for hp in hp_list:
    res_hp = []
    for run in range(num_runs):
        input_file_hp = f"results/ideal/{hp}_{run}_hp.json"
        with open(input_file_hp, 'r') as f:
            data = json.load(f)
            res_hp.append(float(data['throughput']))
    for be in be_list:
        df_hp_ideal_throughput.at[be, hp] = f"{round(np.average(res_hp),2)}/{round(np.std(res_hp),2)}"

for be in be_list:
    res_be = []
    for run in range(num_runs):
        input_file_be = f"results/ideal/{be}_{run}_be.json"
        with open(input_file_be, 'r') as f:
            data = json.load(f)
            res_be.append(float(data['throughput']))
    for hp in hp_list:
        df_be_ideal_throughput.at[be, hp] = f"{round(np.average(res_be),2)}/{round(np.std(res_be),2)}"

df_hp_ideal_throughput.to_csv(f'results/inf_throughput_ideal.csv')
df_be_ideal_throughput.to_csv(f'results/train_throughput_ideal.csv')


# for baseline in baselines[:-1]:
#     df_hp_throughput = pd.DataFrame(0.0, index=models, columns=models)
#     df_be_throughput = pd.DataFrame(0.0, index=models, columns=models)
#     for be,hp in zip(be_list, hp_list):
#         res_hp = []
#         res_be = []
#         for run in range(num_runs):
#             input_file_hp = f"results/{baseline}/{be}_{hp}_{run}_hp.json"
#             with open(input_file_hp, 'r') as f:
#                 data = json.load(f)
#                 res_hp.append(float(data['throughput']))

#             input_file_be = f"results/{baseline}/{be}_{hp}_{run}_be.json"
#             with open(input_file_be, 'r') as f:
#                 data = json.load(f)
#                 res_be.append(float(data['throughput']))

#         df_hp_throughput.at[be, hp] = f"{round(np.average(res_hp),2)}/{round(np.std(res_hp),2)}"
#         df_be_throughput.at[be, hp] = f"{round(np.average(res_be),2)}/{round(np.std(res_be),2)}"

#     df_hp_throughput.to_csv(f'results/train_throughput_{baseline}.csv')
#     df_be_throughput.to_csv(f'results/inf_throughput_{baseline}.csv')
