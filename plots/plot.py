# %%
import matplotlib.pyplot as plt
import numpy as np

model2id = {
    'ResNet50': 0,
    'MobileNetV2': 1,
    'ResNet101': 2,
    'BERT': 3,
    'Transformer': 4
}

id2model = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT', 'Transformer']

p50_data = {
    ('ResNet50', 'ResNet50'): {
        'Sequential': [12.6, 12.6],
        'Streams': [10.7, 10.8],
        'ORION': [9.4, 9.3],
        'MPS': [9.94, 9.94]
    },
    ('ResNet50', 'MobileNetV2'): {
        'Sequential': [10.0, 8.6],
        'Streams': [9.8, 8.3],
        'ORION': [9, 7.5],
        'MPS': [8.01, 4.91]
    },
    ('ResNet50', 'ResNet101'): {
        'Sequential': [12.2, 19.6],
        'Streams': [10, 18.2],
        'ORION': [9, 17],
        'MPS': [9.51, 17.15]
    },
    ('ResNet50', 'BERT'): {
        'Sequential': [56.1, 55.9],
        'Streams': [21.9, 60.8],
        'ORION': [19.2, 60.3],
        'MPS': [21.45, 59.11]
    },
    ('ResNet50', 'Transformer'): {
        'Sequential': [14.2, 28],
        'Streams': [13, 24],
        'ORION': [11, 23],
        'MPS': [13.62, 25.44]
    },
    ('MobileNetV2', 'MobileNetV2'): {
        'Sequential': [8.3, 8.4],
        'Streams': [8.1, 8.2],
        'ORION': [7.56, 7.5],
        'MPS': [4.84, 4.84]
    },
    ('MobileNetV2', 'ResNet101'): {
        'Sequential': [8.1, 18.2],
        'Streams': [8.2, 18.5],
        'ORION': [7.22, 17.1],
        'MPS': [4.91, 13.86]
    },
    ('MobileNetV2', 'BERT'): {
        'Sequential': [51, 51],
        'Streams': [11.1, 55.4],
        'ORION': [10.4, 54.6],
        'MPS': [10.68, 53.59]
    },
    ('MobileNetV2', 'Transformer'): {
        'Sequential': [8.2, 23],
        'Streams': [7.9, 22.3],
        'ORION': [6.77, 20.66],
        'MPS': [5.86, 22.05]
    },
    ('ResNet101', 'ResNet101'): {
        'Sequential': [21.1, 21.1],
        'Streams': [17.8, 17.9],
        'ORION': [7.56, 7.5],
        'MPS': [16.85, 16.96]
    },
    ('ResNet101', 'BERT'): {
        'Sequential': [58, 60],
        'Streams': [40, 60],
        'ORION': [35, 60],
        'MPS': [38.2, 58.82]
    },
    ('ResNet101', 'Transformer'): {
        'Sequential': [27.2, 26.5],
        'Streams': [24.1, 24.1],
        'ORION': [21.2, 22.2],
        'MPS': [25.18, 25.16]
    },
    ('BERT', 'BERT'): {
        'Sequential': [99.3, 99.1],
        'Streams': [93.2, 92.3],
        'ORION': [89.7, 89.8],
        'MPS': [92.48, 92.48]
    },
    ('BERT', 'Transformer'): {
        'Sequential': [64.5, 63.8],
        'Streams': [67, 45],
        'ORION': [63.1, 44],
        'MPS': [67.32, 47.91]
    },
    ('Transformer', 'Transformer'): {
        'Sequential': [31.4, 31.4],
        'Streams': [28.4, 28.4],
        'ORION': [26.1, 26.1],
        'MPS': [31.62, 31.65]
    },
}

p95_data = {
    ('ResNet50', 'ResNet50'): {
        'Sequential': [13.2, 13.3],
        'Streams': [11.9, 11.8],
        'ORION': [10.3, 10.4],
        'MPS': [10.38, 10.36]
    },
    ('ResNet50', 'MobileNetV2'): {
        'Sequential': [10.7, 9.3],
        'Streams': [10.6, 9],
        'ORION': [9.9, 8.3],
        'MPS': [8.84, 5.43]
    },
    ('ResNet50', 'ResNet101'): {
        'Sequential': [14, 20],
        'Streams': [10.9, 19.5],
        'ORION': [10.3, 18],
        'MPS': [10.36, 18.11]
    },
    ('ResNet50', 'BERT'): {
        'Sequential': [56.1, 57],
        'Streams': [23.1, 61.7],
        'ORION': [23.6, 61.5],
        'MPS': [22.58, 59.85]
    },
    ('ResNet50', 'Transformer'): {
        'Sequential': [15, 29.9],
        'Streams': [14, 25],
        'ORION': [12, 23],
        'MPS': [14.87, 26.27]
    },
    ('MobileNetV2', 'MobileNetV2'): {
        'Sequential': [8.9, 8.9],
        'Streams': [8.8, 8.9],
        'ORION': [8, 7.99],
        'MPS': [5.08, 5.06]
    },
    ('MobileNetV2', 'ResNet101'): {
        'Sequential': [8.7, 19.2],
        'Streams': [9.8, 19.2],
        'ORION': [8.17, 18.3],
        'MPS': [5.32, 14.7]
    },
    ('MobileNetV2', 'BERT'): {
        'Sequential': [52, 67],
        'Streams': [11.8, 55.9],
        'ORION': [12.8, 55.7],
        'MPS': [11.15, 54.05]
    },
    ('MobileNetV2', 'Transformer'): {
        'Sequential': [8.8, 25],
        'Streams': [8.4, 24.5],
        'ORION': [7.42, 21.7],
        'MPS': [6.68, 22.68]
    },
    ('ResNet101', 'ResNet101'): {
        'Sequential': [21.4, 21.4],
        'Streams': [18.9, 18.9],
        'ORION': [17.8, 17.5],
        'MPS': [17.67, 18.0]
    },
    ('ResNet101', 'BERT'): {
        'Sequential': [59, 60],
        'Streams': [42, 67],
        'ORION': [39, 61],
        'MPS': [40.26, 59.54]
    },
    ('ResNet101', 'Transformer'): {
        'Sequential': [27.2, 26.6],
        'Streams': [25.3, 24.8],
        'ORION': [23.2, 24.1],
        'MPS': [26.19, 25.84]
    },
    ('BERT', 'BERT'): {
        'Sequential': [99.5, 101],
        'Streams': [93.9, 93.8],
        'ORION': [90.2, 90.3],
        'MPS': [92.7, 92.7]
    },
    ('BERT', 'Transformer'): {
        'Sequential': [64.9, 66.8],
        'Streams': [68, 47],
        'ORION': [65.7, 45],
        'MPS': [68.56, 49.39]
    },
    ('Transformer', 'Transformer'): {
        'Sequential': [32.1, 32],
        'Streams': [29, 29],
        'ORION': [28, 28],
        'MPS': [32.41, 32.28]
    },
}


train_train_data = {
    ('ResNet50', 'ResNet50'): {
        'Sequential': [27.9, 27.89],
        'Streams': [50.27, 50.27],
        'MPS': [49.85, 49.81]
    },
    ('ResNet50', 'MobileNetV2'): {
        'Sequential': [37.54, 30.29],
        'Streams': [61.26, 59.49],
        'MPS': [60.02, 60.44]
    },
    ('ResNet50', 'ResNet101'): {
        'Sequential': [37.51, 45.27],
        'Streams': [71.40, 76.06],
        'MPS': [61.92, 75.32]
    },
    ('ResNet50', 'BERT'): {
        'Sequential': [66.41, 50.5],
        'Streams': [105.27, 98.23],
        'MPS': [103.23, 95.58]
    },
    ('ResNet50', 'Transformer'): {
        'Sequential': [95.12, 51.13],
        'Streams': [129.35, 126.74],
        'MPS': [126.43, 122.95]
    },
    ('MobileNetV2', 'MobileNetV2'): {
        'Sequential': [22.53, 22.54],
        'Streams': [40.28, 40.32],
        'MPS': [39.58, 39.54]
    },
    ('MobileNetV2', 'ResNet101'): {
        'Sequential': [38.04, 45.22],
        'Streams': [69.28, 75.65],
        'MPS': [71.88, 74.35]
    },
    ('MobileNetV2', 'BERT'): {
        'Sequential': [61.29, 60.94],
        'Streams': [108.03, 100.5],
        'MPS': [105.82, 89.47]
    },
    ('MobileNetV2', 'Transformer'): {
        'Sequential': [76.79, 51.22],
        'Streams': [112.46, 104.45],
        'MPS': [109.23, 85.16]
    },
    ('ResNet101', 'ResNet101'): {
        'Sequential': [45.26, 45.28],
        'Streams': [81.99, 81.99],
        'MPS': [80.96, 81.06]
    },
    ('ResNet101', 'BERT'): {
        'Sequential': [76.4, 59.99],
        'Streams': [123.92, 109.87],
        'MPS': [120.82, 102.9]
    },
    ('ResNet101', 'Transformer'): {
        'Sequential': [68.88, 51.54],
        'Streams': [98.27, 107.03],
        'MPS': [97.41, 104.12]
    },
    ('BERT', 'BERT'): {
        'Sequential': [59.97, 60.0],
        'Streams': [116.01, 116.01],
        'MPS': [104.3, 104.29]
    },
    ('BERT', 'Transformer'): {
        'Sequential': [70.46, 60.42],
        'Streams': [102.24, 116.47],
        'MPS': [96.53, 108.73]
    },
    ('Transformer', 'Transformer'): {
        'Sequential': [50.89, 50.89],
        'Streams': [91.39, 91.36],
        'MPS': [81.06, 81.07]
    },
}

# %%
model_pair_to_num_iters_train_train = {
    ('ResNet50', 'ResNet50'): (300, 300),
    ('ResNet50', 'MobileNetV2'): (400, 400),
    ('ResNet50', 'ResNet101'): (400, 300),
    ('ResNet50', 'BERT'): (700, 250),
    ('ResNet50', 'Transformer'): (1000, 300),

    ('MobileNetV2', 'MobileNetV2'): (300, 300),
    ('MobileNetV2', 'ResNet101'): (500, 300),
    ('MobileNetV2', 'BERT'): (800, 300),
    ('MobileNetV2', 'Transformer'): (1000, 300),

    ('ResNet101', 'ResNet101'): (300, 300),
    ('ResNet101', 'BERT'): (500, 300),
    ('ResNet101', 'Transformer'): (450, 300),

    ('BERT', 'BERT'): (300, 300),
    ('BERT', 'Transformer'): (345, 350),
    ('Transformer', 'Transformer'): (300, 300)
}

# %%
transpose = True
num_models = len(id2model)
grid_label_size = 24
xlabel_size = 16
width = 0.2  # the width of the bars
fig, axes = plt.subplots(figsize=(25, 15), ncols=5, nrows=5)
x = np.arange(2)
ymax = 120
color_map = {
    'Sequential': 'b',
    'Streams': 'y',
    'ORION': 'm',
    'MPS': 'g'
}


def plot_p50_latency():
    for i in range(num_models):
        for j in range(num_models):
            ax = axes[num_models - 1 - i, num_models - 1 - j]
            if i > j:
                ax.axis('off')
            else:
                sub_data = p50_data[(id2model[i], id2model[j])]
                for key_id, key in enumerate(sub_data.keys()):
                    offset = width * (key_id - 2)
                    rects = ax.bar(x + offset, sub_data[key], width, label=key, color=color_map[key])
                    # ax.bar_label(rects, padding=1)
                    ax.set_xticks(ticks=[0, 1], labels=[id2model[i], id2model[j]], fontsize=15)
    for ax, col in zip(axes[num_models - 1], id2model[::-1]):
        ax.set_title(col, y=-0.2, pad=-16, fontsize=grid_label_size)

    for ax, row in zip(axes[:, 0], id2model[::-1]):
        ax.set_ylabel(row, fontsize=grid_label_size)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 25})
    fig.suptitle('P50 latency', fontsize=32)
    plt.show()

def plot_durations():
    for i in range(num_models):
        for j in range(num_models):
            ax  = axes[num_models - 1 - i, num_models - 1 - j]
            if i > j:
                ax.axis('off')
            else:
                sub_data = train_train_data[(id2model[i], id2model[j])]
                for key_id, key in enumerate(sub_data.keys()):
                    offset = width * (key_id-1)
                    iterations0, iterations1 = model_pair_to_num_iters_train_train[(id2model[i], id2model[j])]
                    # subtract warm up iterations
                    iterations0 = iterations0 - 10
                    iterations1 = iterations1 - 10
                    duration0, duration1 = sub_data[key]
                    if key in ['MPS', 'Streams']:
                        data_to_plot = [iterations0/duration0, iterations1/duration1]
                    elif key == 'Sequential':
                        # add penalty
                        duration0 += duration1/2
                        duration1 += duration0/2
                        data_to_plot = [iterations0 / duration0, iterations1 / duration1]

                    rects = ax.bar(x + offset, data_to_plot, width, label=key, color=color_map[key])
                    # ax.bar_label(rects, padding=1)
                    ax.set_xticks(ticks=[0, 1], labels=[id2model[i], id2model[j]], fontsize=15)


    for ax, col in zip(axes[num_models - 1], id2model[::-1]):
        ax.set_title(col, y=-0.2,pad=-16, fontsize=grid_label_size)

    for ax, row in zip(axes[:, 0], id2model[::-1]):
        ax.set_ylabel(row, fontsize=grid_label_size)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 25})
    fig.suptitle('Throughput (iterations per second)', fontsize=32)
    plt.show()


# %%
plot_durations()