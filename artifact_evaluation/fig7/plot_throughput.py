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

def get_data(csv_files, error=False):

    df_train_input = pd.read_csv(csv_files[0])
    df_train_input = df_train_input.drop(df_train_input.columns[0], axis=1)
    df_train_input.index = models

    df_inf_input = pd.read_csv(csv_files[1])
    df_inf_input = df_inf_input.drop(df_inf_input.columns[0], axis=1)
    df_inf_input.index = models

    df_train = pd.DataFrame()
    df_inf_new = pd.DataFrame()

    for model_row in models:
        for model_col in models[:2]:
            cell_train = df_train_input.at[model_row, model_col]
            cell_inf = df_inf_input.at[model_row, model_col]

            df_train.at[model_row, model_col] = float(cell_train.split('/')[0])
            df_inf_new.at[model_row, model_col] = float(cell_inf.split('/')[0])
    if error:
        return df_train.std(), df_inf_new.std()
    else:
        return df_train.mean(), df_inf_new.mean()

# %%
method2file = {
    'MPS': ['results/train_throughput_mps.csv', 'results/inf_throughput_mps.csv'],
    'REEF policy': ['results/train_throughput_reef.csv', 'results/inf_throughput_reef.csv'],
    'Orion': ['results/train_throughput_orion.csv', 'results/inf_throughput_orion.csv'],
    'Ideal': ['results/train_throughput_ideal.csv', 'results/inf_throughput_ideal.csv'],
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
x = np.arange(len(models[:2]))
colors = ["royalblue", "darkorange", "green", "red", "mediumpurple", "saddlebrown"]

for method_id, method in enumerate(methods):

    ax.bar(
         x + width * method_id, method2data[method][1][:2], width, yerr=method2err[method][1][:2],
        align='edge', hatch="\\", color = colors[method_id],
    )
    ax.bar(
        x + width * method_id, method2data[method][0][:2], width,
        label=method, yerr=method2err[method][0][:2], bottom=method2data[method][1][:2],
        align='edge', hatch="/", color = colors[method_id], alpha=0.6
    )

x_tick_positions = x + width * len(methods) / 2
ax.set_xticks(
    ticks=x_tick_positions,
    labels=models[:2], fontsize=22
)
plt.yticks(fontsize=22)
ax.set_ylabel('Total Throughput (requests/sec)', fontsize=label_font_size)
ax.set_xlabel('High-priority Inference job', fontsize=label_font_size)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 20}, borderaxespad=2)

#plt.show()
plt.savefig("fig7b.png", bbox_inches="tight")
