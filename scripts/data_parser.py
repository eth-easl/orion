# %%
import json
import os
from string import Template
import statistics as stats
import numpy as np
import pandas as pd


models = ['resnet50', 'mobilenet_v2', 'resnet101', 'bert', 'transformer']
models_better_names = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']

model2id = {
    'resnet50': 0,
    'mobilenet_v2': 1,
    'resnet101': 2,
    'bert': 3,
    'transformer': 4
}


directory = '/Users/sherlock/programs/gpu_share_data/inf-train-poisson2/sequential'
file_name_template = Template('log_0_train-${model0}-eval-${model1}-time-slice-dummy-True.log.json')

num_models = len(models)

duration_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p50_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p95_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

iteration_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

for model0_id in range(5):
    for model1_id in range(5):
        model0 = models[model0_id]
        model1 = models[model1_id]

        filename = file_name_template.substitute(
            model0=model0,
            model1=model1,
        )
        file_fullname = os.path.join(directory, filename)

        try:
            with open(file_fullname, 'r') as f:
                data = json.load(f)
        except:
            print(f'{model0} with {model1} cannot be found')
            continue

        duration_df_raw[model0_id][model1_id] = round(data['duration'], 2)
        p50_df_raw[model0_id][model1_id] = round(data['p50-1'], 2)
        p95_df_raw[model0_id][model1_id] = round(data['p95-1'], 2)
        iteration_df_raw[model0_id][model1_id] = round(data['iteration0'], 2)





duration_df = pd.DataFrame(data=duration_df_raw, columns=models_better_names, index=models_better_names)
p50_df = pd.DataFrame(data=p50_df_raw, columns=models_better_names, index=models_better_names)
p95_df = pd.DataFrame(data=p95_df_raw, columns=models_better_names, index=models_better_names)
iteration_df = pd.DataFrame(data=iteration_df_raw, columns=models_better_names, index=models_better_names)


# %%
# num_reqs_df.to_json('./related/Tick-Tock/num_reqs.json', indent=4, orient='index')

p95_df.to_clipboard()

# %%
# for model0_id in range(5):
#     for model1_id in range(5):
#         if model1_id >= model0_id:
#             model0 = models[model0_id]
#             model1 = models[model1_id]
#             filename = file_name_template.substitute(
#                 model0=model0,
#                 model1=model1,
#             )
#             file_fullname = os.path.join(directory, filename)
#
#             try :
#                 with open(file_fullname, 'r') as f:
#                     data = json.load(f)
#             except:
#                 print(f'{model0} with {model1} cannot be found')
#                 continue
#
#             # cell = f"{round(data['p95-0'], 2)}/{round(data['p95-1'], 2)}"
#             cell = f"{round(data['p50-0'], 2)}/{round(data['p50-1'], 2)}"
#             # cell = round(data['duration'], 2)
#
#         else:
#             model0 = models[model1_id]
#             model1 = models[model0_id]
#             filename = file_name_template.substitute(
#                 model0=model0,
#                 model1=model1,
#             )
#             file_fullname = os.path.join(directory, filename)
#
#             try:
#                 with open(file_fullname, 'r') as f:
#                     data = json.load(f)
#             except:
#                 print(f'{model0} with {model1} cannot be found')
#                 continue
#
#             cell = f"{round(data['p50-1'], 2)}/{round(data['p50-0'], 2)}"
#             # cell = f"{round(data['p95-1'], 2)}/{round(data['p95-0'], 2)}"
#             # cell = round(data['duration'], 2)
#         table_df_raw[model0_id][model1_id] = cell