from ctypes import *
import torch
import numpy as np
import os

class PyScheduler:
    
    def __init__(self, sched_lib):
        
        torch.cuda.set_device(0)
        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib

        home_dir = os.path.expanduser('~')
        model_lib_dir = home_dir + "/gpu_share_repo/cpp_backend/model_kernels/"

        self._model_lib = {
                "resnet50": model_lib_dir + "resnet50",
                "resnet101": model_lib_dir + "resnet101",
                "vgg16_bn": model_lib_dir + "vgg16_bn",
                "mobilenet": model_lib_dir + "mobilenet",
                "gnmt": model_lib_dir + "gnmt_50_64",
                "bert": model_lib_dir + "bert_large_8"
        }
    
    def run_scheduler(self, barrier, tids, model_names):
        
        torch.cuda.profiler.cudart().cudaProfilerStart()
        
        model_names_ctypes = [x.encode('utf-8') for x in model_names]
        lib_names = [self._model_lib[x].encode('utf-8') for x in model_names]

        print(model_names, lib_names)

        self._sched_lib.setup(self._scheduler, tids[0], tids[1], model_names_ctypes[0], lib_names[0], model_names_ctypes[1], lib_names[1])

        barrier.wait()

        num_clients = 1 if tids[1]==0 else 2
        self._sched_lib.sched_func(self._scheduler, num_clients)

        torch.cuda.synchronize()

