
import json

# %%
with open('scripts/REAL.json') as f:
    data = json.load(f)

tasks = data['tasks']


# %%
traces = [task['load']['trace'] for task in tasks]

# %%
num_traces = [len(trace) for trace in traces]

# %%
import pandas as pd
from matplotlib import pyplot as plt
traces = [pd.Series(trace) for trace in traces]

inter_arrival_times = [trace.diff().dropna() for trace in traces]

# %%
label=[f'Trace {i}' for i in range(len(inter_arrival_times))]
plt.hist(inter_arrival_times, density=True, bins=100, cumulative=True, histtype='step', label=label)
plt.legend()
plt.xlabel('Inter-arrival time (ms)')
plt.ylabel('probability')
plt.title('Cumulative Distribution Function of inter-arrival time from each trace')
plt.show()