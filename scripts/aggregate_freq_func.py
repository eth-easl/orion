# %%
import os
directory = '/Users/sherlock/programs/trace'

func_ids = set()
func_id2range = dict()
for day in range(1, 15):
    day_str = str(day).zfill(2)
    filename = f'invocations_per_function_frequent.d{day_str}.csv'
    print(f'processing day {day_str}...')

    with open(os.path.join(directory, filename), 'r') as f:
        lines = f.readlines()

    lines = lines[1:]

    for line in lines:
        fields = line.strip().split(',')
        func_id = (fields[1], fields[2])
        func_ids.add(func_id)

        vals = [int(cell) for cell in fields[4:]]
        freq_range = (min(vals), max(vals))
        func_id2range[func_id] = freq_range


# print(f'there are {len(func_ids[0].intersection(*func_ids[1:]))} frequent functions')
print(f'freq ranges: {func_id2range.values()}')

# %%
