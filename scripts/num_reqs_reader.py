# %%
import json
import os
from string import Template
import statistics as stats
import numpy as np
import pandas as pd


models = ['resnet50', 'mobilenet_v2', 'resnet101', 'bert', 'transformer']
model2id = {
    'resnet50': 0,
    'mobilenet_v2': 1,
    'resnet101': 2,
    'bert': 3,
    'transformer': 4
}
num_models = len(models)

# %%
reqs0_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

reqs1_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

# %%
num_reqs_df = pd.read_csv('./related/Tick-Tock/num_reqs.csv')
num_reqs_df = num_reqs_df.drop(num_reqs_df.columns[0], axis=1)
num_reqs_df.index = models
num_reqs_df.columns = models
num_reqs_df.to_json('./related/Tick-Tock/num_reqs.json', indent=4, orient='index')
# %%
for model_row in models:
    for model_col in models:
        cell = num_reqs_df.at[model_row, model_col]
        num_reqs0 = int(cell.split('/')[0])
        num_reqs1 = int(cell.split('/')[1])
        reqs0_raw[model2id[model_row]][model2id[model_col]] = num_reqs0
        reqs1_raw[model2id[model_row]][model2id[model_col]] = num_reqs1

# %%
num_reqs0_df = pd.DataFrame(reqs0_raw, index=models, columns=models)
num_reqs1_df = pd.DataFrame(reqs1_raw, index=models, columns=models)
# %%

num_reqs0_df.to_json('./related/Tick-Tock/num_reqs0.json', indent=4, orient='index')
num_reqs1_df.to_json('./related/Tick-Tock/num_reqs1.json', indent=4, orient='index')
