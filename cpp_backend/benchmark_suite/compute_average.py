import sys
import numpy as np

inf_durations={
    "resnet50": 6280,
    "mobilenet": 4940,
    "resnet101": 10500,
    "bert": 49020,
    "transformer": 17000
}

ifile = sys.argv[1]

durations = []
with open(ifile, 'r') as f:
    lines = f.readlines()
    for l in lines[1:]:
        tokens = l.split(",")
        durations.append(float(tokens[-1])/1000)

avg_duration = np.average(np.asarray(durations))
for hp_inference in inf_durations:
    D = (0.1 * inf_durations[hp_inference])/avg_duration
    print(f"{hp_inference}, Average duration: {avg_duration} us, hp duration is {inf_durations[hp_inference]} us, D is {D}")