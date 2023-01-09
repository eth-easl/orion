import argparse
import json
import threading
import time
from ctypes import *

from train_imagenet import imagenet_loop
from scheduler_frontend import PyScheduler
from torchvision import models
import torch

def launch_jobs():

    torch.manual_seed(42)


    # init
    barrier = threading.Barrier(2)
    #cu_lib = cdll.LoadLibrary("/home/fot/elastic-spot-ml/scheduling/cpp_backend/cuda_capture/libint.so")
    sched_lib = cdll.LoadLibrary("/home/fot/gpu_share/cpp_backend/scheduler.so")
    py_scheduler = PyScheduler(sched_lib)

    #queue0 = cu_lib.kqueue0
    #mutex0 = cu_lib.mutex

    model = models.__dict__['resnet50'](num_classes=1000)

    model = model.to(0) # to GPU

    print(torch.__version__)

    torch.cuda.synchronize()

    # start threads
    train_thread = threading.Thread(target=imagenet_loop, args=(model, 32, None, 0, barrier, 0))

    sched_thread = threading.Thread(target=py_scheduler.run_scheduler, args=(barrier,))

    train_thread.start()
    sched_thread.start()

    #imagenet_loop(model, data, 32, None, 0, barrier, 0)

    train_thread.join()
    sched_thread.join()

    print("--------- all threads joined!")

if __name__ == "__main__":
    launch_jobs()
