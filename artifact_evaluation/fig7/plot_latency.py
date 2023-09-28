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

def get_data(csv_file, error=False):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    for model_row in models:
        for model_col in models:
            cell = df.at[model_row, model_col]
            df.at[model_row, model_col] = float(cell.split('/')[0]) #float(cell.split('/')[1]) if error else float(cell.split('/')[0])
    if not error:
        return df.mean()

    df = df.std()
    for i in range(len(models)):
        df[i] = min(df[i],300)
    return df



# %%
method2file = {
    'Temporal Sharing': 'results/latency/sequential.csv',
    'Streams': 'results/latency/streams.csv',
    'MPS': 'results/latency/mps.csv',
    'REEF policy': 'results/latency/reef.csv',
    'Orion': 'results/latency/orion.csv',
    'Ideal': 'results/latency/ideal.csv'
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

    bar = ax.bar(
        x + width * method_id, method2data[method], width,
        label=method, yerr=method2err[method],
        align='edge'
    )
    bars.append(bar)

for i,r in enumerate(bars[0]):
    plt.text(r.get_x() + r.get_width()/2.0, 300, f"{method2data['Temporal Sharing'][i]:.0f}", ha='center', va='bottom', fontsize=13)
    #print(r.get_height())

x_tick_positions = x + width * len(methods) / 2
ax.set_xticks(
    ticks=x_tick_positions,
    labels=models, fontsize=22
)
plt.yticks(fontsize=22)
ax.set_ylim(0,300)
ax.set_ylabel('Average p95 inference latency (ms)', fontsize=label_font_size)
ax.set_xlabel('High-priority inference job', fontsize=label_font_size)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.08),ncols=6, fontsize=18)

#plt.show()
plt.savefig("fig7a.png", bbox_inches="tight")