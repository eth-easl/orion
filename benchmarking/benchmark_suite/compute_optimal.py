import sys

ifile = sys.argv[1]
dur_total = 0

with open(ifile, 'r') as f:
    lines = f.readlines()
    for l in lines[1:]:
        tokens = l.split(",")
        sms_used = int(tokens[-2])
        dur = float(tokens[-1])/1000
        if (sms_used>80):
            dur_total += dur

dur_total_ms = dur_total/1000
print(dur_total_ms*2)
