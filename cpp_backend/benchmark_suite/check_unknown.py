import sys
import numpy as np

ifile = sys.argv[1]

sms_used = []
durations = []
durations_all = []
with open(ifile, 'r') as f:
    lines = f.readlines()
    for l in lines[1:]:
        tokens = l.split(",")
        profile = int(tokens[1])
        if profile==-1:
            sms_used.append(int(tokens[-2]))
            durations.append(float(tokens[-1])/1000)
        durations_all.append(float(tokens[-1])/1000)

np.set_printoptions(threshold=np.inf)
print(np.sort(sms_used))
print(np.sort(durations))
print(len(sms_used)/len(durations_all))
#print(f"average: {np.average(np.asarray(durations))}")
