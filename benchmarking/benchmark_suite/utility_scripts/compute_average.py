import sys
import numpy as np

inf_durations={
    #"resnet50": 6280,
    # "mobilenet": 4940,
    # "resnet101": 10500,
    # "bert": 49020,
    "transformer": 17000
}

ifile = sys.argv[1]

durations = []
sms_used = []

in_between = []

with open(ifile, 'r') as f:
    lines = f.readlines()
    for l in lines[1:]:
        tokens = l.split(",")
        sms_used.append(int(tokens[-2]))
        dur = float(tokens[-1])/1000
        durations.append(dur)
        if (dur>=320 and dur<=350):
            print(l)
            in_between.append(dur)

avg_duration = np.average(np.asarray(durations))
max_duration = max(durations)
# print(np.sort(durations))
# print(np.sort(sms_used))

p50 = np.percentile(durations, 50)
p75 = np.percentile(durations, 75)
p95 = np.percentile(durations, 95)
p99 = np.percentile(durations, 99)

print(len(in_between))

for hp_inference in inf_durations:
    D = (0.025 * inf_durations[hp_inference])/avg_duration
    print(f"{hp_inference}, Average duration: {avg_duration} us, max duration is {max_duration} us, hp duration is {inf_durations[hp_inference]} us, D is {D}")