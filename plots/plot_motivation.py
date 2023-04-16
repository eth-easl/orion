# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
# This script corresponds to figure 2

# %%
# data = {
#     'Ideal': np.array([
#         [157.14, 93.43],
#         [157.22, 12.88],
#         [12.89, 12.88]
#     ]),
#     'MPS': np.array([
#         [98.80, 53.89],
#         [53.47, 10.77],
#         [7.33, 7.33]
#     ]),
#     'Streams': np.array([
#         [106, 57.57],
#         [32.52, 11.25],
#         [7.19, 7.19]
#     ])
# }

data = {
    'inference-inference': {
        'model0': [157.14, 98.80, 106],
        'model1': [93.43, 53.89, 57.57],
        'methods': ['Ideal', 'MPS', 'Streams']
    },
    'inference-training': {
        'model0': [157.22, 53.47, 32.52],
        'model1': [12.88, 10.77, 11.25],
        'methods': ['Ideal', 'MPS', 'Streams'],
    },
    'training-training': {
        'model0': [12.89, 7.33, 7.19, 6.67],
        'model1': [12.88, 7.33, 7.19, 6.67],
        'methods': ['Ideal', 'MPS', 'Streams', 'Tick-Tock'],
    }
}

# %%


bottom = np.zeros(3)
x = np.arange(3)
bar_width = 0.4
legend_size = 22
axis_label_font_size = 25
bar_distance = 0.4
edge_color = 'black'
hatch = 'xx'
labelpad = 15
colors = {
    'Ideal': (0.431, 0.78, 0.902),
    'MPS': (0.431, 0.902, 0.478),
    'Streams': (0.902, 0.894, 0.431),
    'Tick-Tock': (0.871, 0.424, 0.878)
}

fig, ax = plt.subplots(figsize=(14, 8))
x_tick_positions = []
for combi_id, combination in enumerate(data.keys()):
    combi_data = data[combination]

    num_methods = len(combi_data['methods'])

    x_positions = np.linspace(0, bar_distance * num_methods, num_methods, endpoint=False)
    offset = combi_id * 2
    methods = combi_data['methods']
    x_tick_positions.append((x_positions[num_methods - 1] + bar_width)/2 + offset)
    ax.bar(
        x_positions + offset, combi_data['model0'], bar_width,
        label=methods if combi_id == 2 else '',
        color=[colors[method] for method in methods],
        edgecolor=edge_color,
        align='edge'
    )

    ax.bar(
        x_positions + offset, combi_data['model1'], bar_width,
        # label=[method + ': model 1' for method in methods] if combi_id == 2 else '',
        bottom=combi_data['model0'],
        color=colors.values(),
        hatch=hatch,
        edgecolor=edge_color,
        align='edge'
    )

# ax.tick_params(axis='y', size=15)
ax.set_xticks(
    ticks=x_tick_positions,
    labels=data.keys(), fontsize=axis_label_font_size
)
ax.tick_params(axis='x', pad=labelpad)

ax.set_ylabel('Throughput (batch/sec)', fontsize=axis_label_font_size, labelpad=labelpad)

solid_artist = mpatches.Patch(
    facecolor='white',
    edgecolor=edge_color,
)

shaded_artist = mpatches.Patch(
    facecolor='white',
    edgecolor=edge_color,
    hatch=hatch,
)

plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
handles.extend([solid_artist, shaded_artist])
labels.extend(['Job 1', 'Job 2'])
fig.legend(
    handles, labels,
    loc='upper right',
    ncols=1,
    borderaxespad=2,
    prop={'size': legend_size},
)
plt.show()

