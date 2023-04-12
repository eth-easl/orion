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

# %%
width = 0.2  # the width of the bars
fig, axes = plt.subplots(figsize=(20, 10), ncols=5, nrows=5)
x = np.arange(2)
for i in range(5):
    for j in range(5):
        if i > j:
            # axes[i, j].axis('off')
            pass
        else:
            ax = axes[i, j]
            sub_data = p50_data[(id2model[i], id2model[j])]
            for key_id, key in enumerate(sub_data.keys()):
                offset = width * (key_id-2)
                rects = ax.bar(x + offset, sub_data[key], width, label=key)
                ax.bar(x + offset, sub_data[key], width)
                ax.set_xticks(ticks=[0, 1], labels=[id2model[i], id2model[j]])

for ax, col in zip(axes[0], id2model):
    ax.set_title(col)

for ax, row in zip(axes[:, 0], id2model):
    ax.set_ylabel(row, size='large')


fig.suptitle('P50 latency', fontsize=32)
plt.show()