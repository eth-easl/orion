# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']
model2id = {
    'ResNet50': 0,
    'MobileNetV2': 1,
    'ResNet101': 2,
    'BERT': 3,
    'Transformer': 4
}

request_rates = {
    'ResNet50': 50,
    'MobilenetV2': 65,
    'ResNet101': 25,
    'BERT': 5,
    'Transformer': 12
}

batch_sizes = {
    'ResNet50': 4,
    'MobileNetV2': 4,
    'ResNet101': 4,
    'BERT': 2,
    'Transformer': 4
}

# %%

def process_num_reqs(csv_file):
    '''
    Return entire number of batches.
    Assume originally each cell is val1/val2, this function returns a Dataframe whose cell is val1 + val2,
    because we care about the total throughput.
    '''
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    for model_row in models:
        for model_col in models:
            cell = df.at[model_row, model_col]
            num_reqs0 = int(cell.split('/')[0])
            num_reqs1 = int(cell.split('/')[1])
            df.at[model_row, model_col] = num_reqs0 + num_reqs1 - 20 # warm up
    return df

# %%
def average_throughput(csv_file, workload_df):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    for model_row in models:
        for model_col in models:
            cell = df.at[model_row, model_col]
            df.at[model_row, model_col] = workload_df.at[model_row, model_col] / cell
    return df.mean()


# %%
workload_df = process_num_reqs('num_reqs.csv')

# %%
# these are the data for "Time fo all requests"
method2file = {
    'Sequential': 'Sequential_time.csv',
    'Streams': 'Streams_time.csv',
    'MPS': 'MPS_time.csv',
    'REEF': 'REEF_time.csv',
    'Orion': 'Orion_time.csv'
}

label_font_size = 20
methods = list(method2file.keys())

method2data = {}

for method, file in method2file.items():
    method2data[method] = average_throughput(file, workload_df)


# %%
# methods.append('Ideal')
# sending {num_reqs_alone} requests, how long is the entire duration, for each model.
single_total_time = [19.25, 14.8, 38.49, 192.45, 80.19]
num_reqs_alone = 1000
single_throughput = pd.Series(single_total_time, index=models)
single_throughput = (num_reqs_alone - 10) / single_throughput # 10 is warm-up


ideal_throughput = [(single_throughput + single_throughput[model]).mean() for model in models]
ideal_throughput = pd.Series(ideal_throughput, index=models)

methods.append('Ideal')
method2data['Ideal'] = ideal_throughput

width = 0.15
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(models))
for method_id, method in enumerate(methods):

    ax.bar(
        x + width * method_id, method2data[method], width,
        label=method,
        align='edge'
    )

x_tick_positions = x + width * len(methods) / 2
ax.set_xticks(
    ticks=x_tick_positions,
    labels=models, fontsize=15
)
ax.set_ylabel('Average throughput (requests/sec)', fontsize=label_font_size)
ax.set_xlabel('High-priority inference job', fontsize=label_font_size)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 20}, borderaxespad=2)

plt.show()