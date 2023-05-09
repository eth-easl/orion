
import json

# %%
with open('scripts/REAL.json') as f:
    data = json.load(f)

tasks = data['tasks']


# %%
raw_traces = [task['load']['trace'] for task in tasks if task['load']['type'] == 'trace']

# %%
# prepend a "0" to the beginning of each trace
traces = [[0] + trace for trace in raw_traces]
num_traces = [len(trace) for trace in raw_traces]

# %%
import pandas as pd
pd_traces = [pd.Series(trace, dtype=int) for trace in traces]

inter_arrival_times = [trace.diff().dropna().to_list() for trace in pd_traces]

# %%
entire_inter_arrival_times = [int(item) for sublist in inter_arrival_times for item in sublist]
with open('scripts/inter_arrival_times.json', 'w') as f:
    json.dump(entire_inter_arrival_times, f)