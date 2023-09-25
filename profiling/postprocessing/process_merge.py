import sys

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]

with open(file1, 'r') as f1:
    lines1 = f1.readlines()
    data1 = lines1[:240]

with open(file2, 'r') as f2:
    lines=f2.readlines()
    data2 = lines[239:]

for d in data2:
    if 'Batch' in d:
        data1 += "BatchNorm,-1,1,1,7\n"
    elif 'Convolution' in d:
        data1 += "Conv,-1,1,1,7\n"
    else:
        data1 += "void at::native::vectorized_elementwise_kernel,-1,1,1,7\n"

with open(file3, 'w') as f3:
    for line in data1:
        f3.write(line)
