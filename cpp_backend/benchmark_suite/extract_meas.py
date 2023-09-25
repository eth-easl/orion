import sys

input_file = sys.argv[1]

i=0
ar = [0, 0]
ar_str = []


# with open(input_file, 'r') as f:
#      while (1):
#          l = f.readline()
#          i+=1
#          print(i, l)


tt = []
iters = []
with open(input_file, 'r') as f:
    lines = f.readlines()
    for i,l in enumerate(lines):

        # if 'p50' in l:
        #       tokens = l.split(",")
        #       #print(l)
        #       if "Client 0" in tokens[0]:
        #           ar[0] = round(float(tokens[2].split(" ")[-2])*1000, 2)
        #       else:
        #           ar[1] = round(float(tokens[2].split(" ")[-2])*1000, 2)
        #       i += 1
        #       if (i%1==0):
        #           #s = f"{ar[0]}/{ar[1]}"
        #           s = f"{ar[1]}"
        #           ar_str.append(s)
        #       if (i==10):
        #           i=0

        if 'Total loop' in l and 'Client' not in l:
             tokens = l.split(" ")
             tt.append(round(float(tokens[-2]),2))
        if '=======' in l:
             tokens = l.split(" ")
             iters.append(int(tokens[-2]))
print(len(tt))

#for i in range(5):
#      print(f"{ar_str[5*i]},{ar_str[5*i+1]},{ar_str[5*i+2]},{ar_str[5*i+3]},{ar_str[5*i+4]}")

# for i in range(5):
#     print(f"{tt[5*i]},{tt[5*i+1]},{tt[5*i+2]},{tt[5*i+3]},{tt[5*i+4]}")

# for i in range(5):
#      print(f"{iters[5*i]},{iters[5*i+1]},{iters[5*i+2]},{iters[5*i+3]},{iters[5*i+4]}")

inf_requests = [9200, 12000, 5500, 1200, 3400]

#print("--------- High Priority Throughput:")
# hp_th = []
# for i in range(len(tt)):
#       hp_th.append(round(inf_requests[i%5]/tt[i],2))
# for i in range(5):
#        print(f"{hp_th[5*i]},{hp_th[5*i+1]},{hp_th[5*i+2]},{hp_th[5*i+3]},{hp_th[5*i+4]}")

# #print("--------- Low Priority Throughput:")

hp_th = []
for i in range(len(tt)):
     hp_th.append(round(iters[i]/tt[i],2))
for i in range(5):
     print(f"{iters[5*i]},{iters[5*i+1]},{iters[5*i+2]},{iters[5*i+3]},{iters[5*i+4]}")


# print("--------- High Priority Throughput:")
# hp_th = []
# for i in range(len(tt)):
#     hp_th.append(round(1000/tt[i],2))
# for i in range(5):
#      print(f"{hp_th[5*i]},{hp_th[5*i+1]},{hp_th[5*i+2]},{hp_th[5*i+3]},{hp_th[5*i+4]}")

# print("--------- Low Priority Throughput:")
# hp_th = []
# for i in range(len(tt)):
#     hp_th.append(round(iters[i]/tt[i],2))
# for i in range(5):
#      print(f"{hp_th[5*i]},{hp_th[5*i+1]},{hp_th[5*i+2]},{hp_th[5*i+3]},{hp_th[5*i+4]}")


print("--------- Total Throughput:")
hp_th = []
for i in range(len(tt)):
    hp_th.append(round((iters[i]+1000)/tt[i],2))
for i in range(5):
     print(f"{hp_th[5*i]},{hp_th[5*i+1]},{hp_th[5*i+2]},{hp_th[5*i+3]},{hp_th[5*i+4]}")
