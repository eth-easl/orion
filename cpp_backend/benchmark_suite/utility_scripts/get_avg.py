import numpy as np

l1 = [[0,0,0,0,0] for _ in range(5)]
l2 = [[0,0,0,0,0] for _ in range(5)]

with open('case1', 'r') as f1:
    f1.readline()
    #f1.readline()
    l = f1.readlines()
    for i in range(5):
        t = l[i]
        s = t.split(",")
        for j in range(5):
            l1[i][j] = float(s[j])

with open('case2', 'r') as f2:
    f2.readline()
    #f2.readline()
    l = f2.readlines()

    for i in range(5):
        t = l[i]
        s = t.split(",")
        for j in range(5):
            l2[i][j] = float(s[j])
print(l1, l2)

l_total = [[0,0,0,0,0] for _ in range(5)]
l_std =  [[0,0,0,0,0] for _ in range(5)]
l_str = [[0,0,0,0,0] for _ in range(5)]

for i in range(5):
    for j in range(5):
        l_total[i][j] = round(np.average([l1[i][j],l2[i][j]]),2)
        l_std[i][j] = round(np.std([l1[i][j],l2[i][j]]),2)
        l_str[i][j] = str(l_total[i][j]) + "/" + str(l_std[i][j])
for i in range(5):
    print(f"{l_total[i][0]}/{l_std[i][0]},{l_total[i][1]}/{l_std[i][1]},{l_total[i][2]}/{l_std[i][2]},{l_total[i][3]}/{l_std[i][3]},{l_total[i][4]}/{l_std[i][4]}")
