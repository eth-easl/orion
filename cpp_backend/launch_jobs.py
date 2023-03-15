import argparse
import json
import threading
import time
from ctypes import *
import os
import sys
from torchvision import models
import torch

sys.path.insert(0, "/home/image-varuna/DeepLearningExamples/PyTorch/Translation/GNMT")
from benchmark_suite.gnmt_trainer import gnmt_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
from benchmark_suite.bert_trainer import bert_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer import transformer_loop
sys.path.append("/home/image-varuna/mlcommons/single_stage_detector/ssd")
from benchmark_suite.retinanet_trainer import retinanet_loop

sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/Recommendation/DLRM")
from benchmark_suite.dlrm_trainer import dlrm_loop

from benchmark_suite.train_imagenet import imagenet_loop
from scheduler_frontend import PyScheduler

def seed_everything(seed: int):
    import random, os
    import numpy as np


    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def launch_jobs():

    # init
    barrier = threading.Barrier(3)
    home_directory = os.path.expanduser( '~' )
    sched_lib = cdll.LoadLibrary(home_directory + "/gpu_share_repo/cpp_backend/scheduler/scheduler.so")
    py_scheduler = PyScheduler(sched_lib)

    print(torch.__version__)

    model_names = ["transformer", "transformer"]

    #torch.cuda.synchronize()

    # start threads
    train_thread_0 = threading.Thread(target=transformer_loop, args=(32, None, 0, barrier, 0))
    train_thread_0.start()

    train_thread_1 = threading.Thread(target=transformer_loop, args=(32, None, 0, barrier, 0))
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
