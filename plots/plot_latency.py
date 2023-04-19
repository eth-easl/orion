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

# %%

def average_latency(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    for model_row in models:
        for model_col in models:
            cell = df.at[model_row, model_col]
            df.at[model_row, model_col] = float(cell.split('/')[1])
    return df.mean()


# %%
method2file = {
    'Sequential': 'sequential_p95.csv',
    'Streams': 'streams_p95.csv',
    'MPS': 'MPS_p95.csv',
    'REEF': 'reef_p95.csv',
    'Orion': 'orion_latency.csv'
}

label_font_size = 20
methods = list(method2file.keys())

method2data = {}

for method, file in method2file.items():
    method2data[method] = average_latency(file)

methods.append('Ideal')
method2data['Ideal'] = pd.Series([6.72, 5.47, 11.27, 49.61, 18.96], index=models)
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
ax.set_ylabel('Average p95 inference latency (ms)', fontsize=label_font_size)
ax.set_xlabel('High-priority inference job', fontsize=label_font_size)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 20}, borderaxespad=2)

plt.show()