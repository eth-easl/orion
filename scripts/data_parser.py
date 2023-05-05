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


directory = '/Users/sherlock/programs/gpu_share_data/train-train-priorities-pinned/tick-tock'
file_name_template = Template('log_0_train-${model0}-train-${model1}-tick-tock.log.json')

num_models = len(models)

total_throughput_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

throughput0_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

throughput1_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

iteration1_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

total_time_df_raw = [
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

        total_time = data['duration']
        total_time_df_raw[model0_id][model1_id] = round(total_time, 2)

        iteration1 = int(data['iteration1'])
        throughput1_raw[model0_id][model1_id] = round((iteration1 - 10) / total_time, 2)
        throughput0_raw[model0_id][model1_id] = round((1000 - 10) / total_time, 2)
        total_throughput_df_raw[model0_id][model1_id] = round((iteration1 + 1000 - 10) / total_time, 2)





total_throughput_df = pd.DataFrame(data=total_throughput_df_raw, columns=models_better_names, index=models_better_names)
throughput0_df = pd.DataFrame(data=throughput0_raw, columns=models_better_names, index=models_better_names)
throughput1_df = pd.DataFrame(data=throughput1_raw, columns=models_better_names, index=models_better_names)
total_time_df = pd.DataFrame(data=total_time_df_raw, columns=models_better_names, index=models_better_names)


# %%
# num_reqs_df.to_json('./related/Tick-Tock/num_reqs.json', indent=4, orient='index')

total_time_df.to_clipboard()

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