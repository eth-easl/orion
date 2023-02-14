import sys

intercepter_ops = []
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'INTERCEPTER-CATCH' in line:
            intercepter_ops.append(line)

for x in intercepter_ops:
    print(x)
