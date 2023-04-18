# %%
import json
import os
from string import Template
import statistics as stats
import numpy as np
import pandas as pd

models = ['resnet50', 'mobilenet_v2', 'resnet101', 'bert', 'transformer']
models_better_names = ['ResNet50', 'MobileNetV2', 'Resnet101', 'BERT', 'Transformer']
model2id = {
    'resnet50': 0,
    'mobilenet_v2': 1,
    'resnet101': 2,
    'bert': 3,
    'transformer': 4
}


directory = '/Users/sherlock/programs/gpu_share_data/inf-train-open-MPS'
file_name_template = Template('log_0_train-${model0}-eval-${model1}-MPS-process-dummy-True.log.json')

num_models = len(models)

num_reqs_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

for model0 in models[:2]:
    for model1 in models:
        filename = file_name_template.substitute(
            model0=model0,
            model1=model1,
        )
        file_fullname = os.path.join(directory, filename)
        with open(file_fullname, 'r') as f:
            data = json.load(f)




        num_reqs = round(data['p95-1'], 2)
        num_reqs_df_raw[model2id[model0]][model2id[model1]] = num_reqs

num_reqs_df = pd.DataFrame(data=num_reqs_df_raw, columns=models_better_names, index=models_better_names)


# %%
# num_reqs_df.to_json('./related/Tick-Tock/num_reqs.json', indent=4, orient='index')

num_reqs_df.to_clipboard()