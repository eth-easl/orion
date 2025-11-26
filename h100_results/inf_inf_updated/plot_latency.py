# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']

# %%

def get_data(csv_file, error=False):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    #df = df.drop(df.columns[-3], axis=1)
    #df = df.drop(df.columns[-2], axis=1)
    #df = df.drop(df.columns[-1], axis=1)

    print(df)
    for model_row in models:
        for model_col in models:
            cell = df.at[model_row, model_col]
            df.at[model_row, model_col] = float(cell.split('/')[0]) #float(cell.split('/')[1]) if error else float(cell.split('/')[0])
    if error:
        return df.std()
    else:
        return df.mean()

# %%
method2file = {
    'Temporal': 'results/temporal_latency.csv',
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
x = np.arange(len(models))
bars = []
for method_id, method in enumerate(methods):

    print(method, method2data[method], method2err[method])
    bar = ax.bar(
        x + width * method_id, method2data[method], width,
        label=method, yerr=method2err[method],
        align='edge'
    )
    bars.append(bar)

x_tick_positions = x + width * len(methods) / 2
ax.set_xticks(
    ticks=x_tick_positions,
    labels=models, fontsize=22
)
plt.yticks(fontsize=22)
ax.set_ylim(0, 1000)
ax.set_ylabel('Average p95 inference latency (ms)', fontsize=label_font_size)
ax.set_xlabel('High-priority inference job', fontsize=label_font_size)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper left', ncol=1, fontsize=20)

plt.savefig("inf_inf_poisson_hp_latency.png", bbox_inches="tight")
