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


directory = '/Users/sherlock/programs/gpu_share_data/inf-inf-trace-high-rps/mps'
file_name_template = Template('log_${start_id}_eval-${model0}-eval-${model1}-MPS.log.json')
mode = 'inf-inf'
num_models = len(models)

total_time_df_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

training_iterations_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p50_df_low_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p95_df_low_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p99_df_low_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p50_df_high_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p95_df_high_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

p99_df_high_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

throughput_high_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

throughput_low_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

throughput_total_raw = [
    [0 for i in range(num_models)] for j in range(num_models)
]

num_reqs = {
    'resnet50': 9200,
    'mobilenet_v2': 12000,
    'resnet101': 5500,
    'bert': 1200,
    'transformer': 3400
}

def get_mean_std(data_array, key):
    size = len(data_array)
    if size == 1:
        return f'{round(data_array[0][key], 2)}'
    data = [d[key] for d in data_array]
    return f'{round(stats.mean(data), 2)}/{round(stats.stdev(data), 2)}'

def get_mean_std_throughput(data_array, high_iter_val = None):
    if high_iter_val is None:
        high_iters = [d['iterations0'] for d in data_array]
    else:
        high_iters = [high_iter_val for d in data_array]
    low_iters = [d['iterations1'] for d in data_array]
    total_time = [d['duration'] for d in data_array]

    num_samples = len(data_array)
    high_tp = [(high_iters[i] - 10)/total_time[i] for i in range(num_samples)]
    low_tp = [(low_iters[i] - 10) / total_time[i] for i in range(num_samples)]
    tp = [high_tp[i] + low_tp[i] for i in range(num_samples)]
    if num_samples == 1:
        return f'{round(high_tp[0], 2)}', f'{round(low_tp[0], 2)}', f'{round(tp[0], 2)}'
    else:
        return f'{round(stats.mean(high_tp), 2)}/{round(stats.stdev(high_tp), 2)}'\
            , f'{round(stats.mean(low_tp), 2)}/{round(stats.stdev(low_tp), 2)}'\
            , f'{round(stats.mean(tp), 2)}/{round(stats.stdev(tp), 2)}'


for model0_id in range(3 if 'trace' in directory else 5):
    for model1_id in range(5):
        model0 = models[model0_id]
        model1 = models[model1_id]

        data_array = []
        for start_id in range(3):
            filename = file_name_template.substitute(
                model0=model0,
                model1=model1,
                start_id=start_id
            )
            file_fullname = os.path.join(directory, filename)
            try:
                with open(file_fullname, 'r') as f:
                    data_array.append(json.load(f))
            except:
                print(f'{filename} cannot be found')
                continue

        if mode == 'inf-inf':
            total_time_df_raw[model1_id][model0_id] = get_mean_std(data_array, 'duration')
            training_iterations_raw[model1_id][model0_id] = get_mean_std(data_array, 'iterations1')

            p50_df_low_raw[model1_id][model0_id] = get_mean_std(data_array, 'p50-1')
            p95_df_low_raw[model1_id][model0_id] = get_mean_std(data_array, 'p95-1')
            p99_df_low_raw[model1_id][model0_id] = get_mean_std(data_array, 'p99-1')

            p50_df_high_raw[model1_id][model0_id] = get_mean_std(data_array, 'p50-0')
            p95_df_high_raw[model1_id][model0_id] = get_mean_std(data_array, 'p95-0')
            p99_df_high_raw[model1_id][model0_id] = get_mean_std(data_array, 'p99-0')
            high, low, total = get_mean_std_throughput(data_array)
            throughput_high_raw[model1_id][model0_id] = high
            throughput_low_raw[model1_id][model0_id] = low
            throughput_total_raw[model1_id][model0_id] = total
        elif mode == 'train-train':
            total_time_df_raw[model1_id][model0_id] = get_mean_std(data_array, 'duration')
            high, low, total = get_mean_std_throughput(data_array)
            throughput_high_raw[model1_id][model0_id] = high
            throughput_low_raw[model1_id][model0_id] = low
            throughput_total_raw[model1_id][model0_id] = total
        else:
            total_time_df_raw[model1_id][model0_id] = get_mean_std(data_array, 'duration')
            training_iterations_raw[model1_id][model0_id] = get_mean_std(data_array, 'iterations1')
            p50_df_high_raw[model1_id][model0_id] = get_mean_std(data_array, 'p50-0')
            p95_df_high_raw[model1_id][model0_id] = get_mean_std(data_array, 'p95-0')
            p99_df_high_raw[model1_id][model0_id] = get_mean_std(data_array, 'p99-0')
            # high_iter_val = num_reqs[model0]
            high, low, total = get_mean_std_throughput(data_array)
            throughput_high_raw[model1_id][model0_id] = high
            throughput_low_raw[model1_id][model0_id] = low
            throughput_total_raw[model1_id][model0_id] = total



# %%
if mode == 'inf-inf':
    total_time_df = pd.DataFrame(data=total_time_df_raw, columns=models_better_names, index=models_better_names)
    training_iterations_df = pd.DataFrame(data=training_iterations_raw, columns=models_better_names, index=models_better_names)
    p50_low_df = pd.DataFrame(data=p50_df_low_raw, columns=models_better_names, index=models_better_names)
    p95_low_df = pd.DataFrame(data=p95_df_low_raw, columns=models_better_names, index=models_better_names)
    p99_low_df = pd.DataFrame(data=p99_df_low_raw, columns=models_better_names, index=models_better_names)
    p50_high_df = pd.DataFrame(data=p50_df_high_raw, columns=models_better_names, index=models_better_names)
    p95_high_df = pd.DataFrame(data=p95_df_high_raw, columns=models_better_names, index=models_better_names)
    p99_high_df = pd.DataFrame(data=p99_df_high_raw, columns=models_better_names, index=models_better_names)
    throughput_high_df = pd.DataFrame(data=throughput_high_raw, columns=models_better_names, index=models_better_names)
    throughput_low_df = pd.DataFrame(data=throughput_low_raw, columns=models_better_names, index=models_better_names)
    throughput_total_df = pd.DataFrame(data=throughput_total_raw, columns=models_better_names, index=models_better_names)
elif mode == 'train-train':
    total_time_df = pd.DataFrame(data=total_time_df_raw, columns=models_better_names, index=models_better_names)
    throughput_high_df = pd.DataFrame(data=throughput_high_raw, columns=models_better_names, index=models_better_names)
    throughput_low_df = pd.DataFrame(data=throughput_low_raw, columns=models_better_names, index=models_better_names)
    throughput_total_df = pd.DataFrame(data=throughput_total_raw, columns=models_better_names, index=models_better_names)
else:
    total_time_df = pd.DataFrame(data=total_time_df_raw, columns=models_better_names, index=models_better_names)
    training_iterations_df = pd.DataFrame(data=training_iterations_raw, columns=models_better_names, index=models_better_names)
    p50_high_df = pd.DataFrame(data=p50_df_high_raw, columns=models_better_names, index=models_better_names)
    p95_high_df = pd.DataFrame(data=p95_df_high_raw, columns=models_better_names, index=models_better_names)
    p99_high_df = pd.DataFrame(data=p99_df_high_raw, columns=models_better_names, index=models_better_names)
    throughput_high_df = pd.DataFrame(data=throughput_high_raw, columns=models_better_names, index=models_better_names)
    throughput_low_df = pd.DataFrame(data=throughput_low_raw, columns=models_better_names, index=models_better_names)
    throughput_total_df = pd.DataFrame(data=throughput_total_raw, columns=models_better_names, index=models_better_names)


# %%
# num_reqs_df.to_json('./related/Tick-Tock/num_reqs.json', indent=4, orient='index')
throughput_high_df.to_clipboard()

