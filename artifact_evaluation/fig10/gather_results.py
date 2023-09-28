import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']
baselines = ['sequential', 'streams', 'mps', 'reef', 'orion', 'ideal']

hp_list = ['ResNet50', 'MobileNetV2', 'ResNet101']
be_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']
num_runs = 3

for baseline in baselines:
    df = pd.DataFrame("0", index=models, columns=models)
    for be,hp in zip(be_list, hp_list):
        results = []
        for run in range(num_runs):
            input_file = f"results/{baseline}/{be}_{hp}_{run}_hp.json"
            with open(input_file, 'r') as f:
                data = json.load(f)
                results.append(float(data['p95_latency']))
        df.at[be, hp] = f"{round(np.average(results),2)}/{round(np.std(results),2)}"
    df.to_csv(f'results/{baseline}.csv')
