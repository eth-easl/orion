import argparse
import json
import threading
import time
from ctypes import *
import os

from train_imagenet import imagenet_loop
from scheduler_frontend import PyScheduler
from torchvision import models
import torch

def launch_jobs():

    torch.manual_seed(42)

    # init
    barrier = threading.Barrier(3)
    home_directory = os.path.expanduser( '~' )
    sched_lib = cdll.LoadLibrary(home_directory + "/gpu_share_repo/cpp_backend/scheduler.so")
    py_scheduler = PyScheduler(sched_lib)

    print(torch.__version__)

    model_names = ["vgg16_bn", "vgg16_bn"]

    #torch.cuda.synchronize()

    # start threads
    train_thread_0 = threading.Thread(target=imagenet_loop, args=(model_names[0], 32, None, 0, barrier, 0))
    train_thread_0.start()

    train_thread_1 = threading.Thread(target=imagenet_loop, args=(model_names[1], 32, None, 0, barrier, 1))
    train_thread_1.start()

    tids = [train_thread_0.native_id, train_thread_1.native_id]
    sched_thread = threading.Thread(target=py_scheduler.run_scheduler, args=(barrier, tids, model_names))

    sched_thread.start()

    train_thread_0.join()
    train_thread_1.join()

    print("train joined!")

    sched_thread.join()
    print("sched joined!")

    print("--------- all threads joined!")

if __name__ == "__main__":
    launch_jobs()
