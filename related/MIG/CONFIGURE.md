### How to colocate applications in an A100 gpu using MIG mode

1. Enable MIG and reboot:
```
sudo nvidia-smi -i 0 -mig 1
sudo reboot
```

2. List the sizes and other configurations of the available GPU instances:
```
sudo nvidia-smi mig -lgip
```
Each instance type has a name (e.g. '2g.10gb') and a profile ID (e.g. 14).


3. Create GPU instances. As an example, we will partition the GPU into 3 instances, one '3g.20gb' (ID 9), and two '2g.10gb' (ID 14).
```
sudo nvidia-smi mig -cgi 9,14,14 -C
```

If partitioning was done sucessfully, we can see the created partitions by running
```
sudo nvidia-smi mig -lgi
```

4. Find the IDs of the created instances (UUID)
```
nvidia-smi -L
```

5. Run 3 processes, each in one of the 3 partitions we created before:
```
CUDA_VISIBLE_DEVICES=UUID1 ./app1 &
CUDA_VISIBLE_DEVICES=UUID2 ./app2 &
CUDA_VISIBLE_DEVICES=UUID3 ./app3 &
```

### Reconfigure

MIG instances cannot be reconfigured on the fly. In order to reconfigure, we have to:
1. Kill all running processes
2. Destroy the existing partitions:
```
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
```
3. Create the new partitions, e.g. :
```
sudo nvidia-smi mig -cgi 9,9 -C
```
4. (Re)start the processes
