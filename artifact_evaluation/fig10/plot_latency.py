# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']

# %%

def get_data(csv_file, error=False):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    df = df.drop(df.columns[-3], axis=1)
    df = df.drop(df.columns[-2], axis=1)
    df = df.drop(df.columns[-1], axis=1)

    for model_row in models:
        for model_col in models[:2]:
            cell = df.at[model_row, model_col]
            df.at[model_row, model_col] = float(cell.split('/')[0]) #float(cell.split('/')[1]) if error else float(cell.split('/')[0])
    if error:
        return df.std()
    else:
        return df.mean()

# %%
method2file = {
    'MPS': 'results/mps_latency.csv',
    'REEF policy': 'results/reef_latency.csv',
    'Orion': 'results/orion_latency.csv',
    'Ideal': 'results/ideal_latency.csv'
}

label_font_size = 22
methods = list(method2file.keys())

method2data = {}
method2err = {}

for method, file in method2file.items():
    method2data[method] = get_data(file)
    method2err[method] = get_data(file, error=True)

width = 0.15
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(2)
bars = []
for method_id, method in enumerate(methods):

    bar = ax.bar(
        x + width * method_id, method2data[method], width,
        label=method, yerr=method2err[method],
        align='edge'
    )
    bars.append(bar)

x_tick_positions = x + width * len(methods) / 2
ax.set_xticks(
    ticks=x_tick_positions,
    labels=models[:2], fontsize=22
)
plt.yticks(fontsize=22)
ax.set_ylabel('Average p95 inference latency (ms)', fontsize=label_font_size)
ax.set_xlabel('High-priority inference job', fontsize=label_font_size)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper left', ncol=1, fontsize=20)

plt.savefig("fig10.png", bbox_inches="tight")
