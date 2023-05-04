# %%
import os
directory = '/Users/sherlock/programs/trace'

func_id2range = dict()

for day in range(1, 15):
    day_str = str(day).zfill(2)
    filename = f'invocations_per_function_md.anon.d{day_str}.csv'
    print(f'processing day {day_str}...')

    with open(os.path.join(directory, filename), 'r') as f:
        lines = f.readlines()

    header = lines[0]
    lines = lines[1:]

    def is_frequent_func(line, threshold=60):

        fields = line.strip().split(',')[4:]
        assert len(fields) == 24 * 60

        for field in fields:
            if int(field) < threshold:
                return False

        return True

    new_func_id2range = dict()
    for line in lines:
        if is_frequent_func(line):
            fields = line.strip().split(',')
            func_id = (fields[1], fields[2])
            if day == 1:
                freqs = [int(cell) for cell in fields[4:]]
                new_func_id2range[func_id] = freqs
            else:
                if func_id in func_id2range:
                    freqs = [int(cell) for cell in fields[4:]]
                    new_func_id2range[func_id] = func_id2range[func_id] + freqs
    func_id2range = new_func_id2range

print(f'there are {len(func_id2range)} frequent functions')



# %%
import statistics as stats
import numpy as np
func_id2median_freq = dict()
for func_id, freqs in func_id2range.items():
    func_id2median_freq[func_id] = stats.median(freqs) / 60

bins = [0, 20, 40, 60, 80, 100, max(func_id2median_freq.values())]
hist, bin_edges = np.histogram(list(func_id2median_freq.values()), bins = bins)
print(hist)
print(bin_edges)
# print(func_id2median_freq.values())
