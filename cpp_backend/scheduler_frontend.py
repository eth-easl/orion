import ctypes
from ctypes import *
import torch
import numpy as np
import os

class PyScheduler:

    def __init__(self, sched_lib, num_clients):

        torch.cuda.set_device(0)
        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib
        self._num_clients = num_clients

        home_dir = os.path.expanduser('~')
        model_lib_dir = home_dir + "/gpu_share_repo/cpp_backend/model_kernels/"

        self._model_lib = {
                "resnet50": model_lib_dir + "resnet50_with_profile",
                "resnet101": model_lib_dir + "resnet101_with_profile",
                "vgg16_bn": model_lib_dir + "vgg16_bn_with_profile",
                "mobilenet": model_lib_dir + "mobilenet_with_profile",
                "gnmt": model_lib_dir + "gnmt_50_64_with_profile",
                "bert": model_lib_dir + "bert_large_8_with_profile",
                "transformer": model_lib_dir + "transformer_xl_32_with_profile",
                "retinanet": model_lib_dir + "retinanet_resnet_8_with_profile",
                "dlrm": model_lib_dir + "dlrm_small_with_profile"
        }

    def run_scheduler(self, barrier, tids, model_names):

        torch.cuda.profiler.cudart().cudaProfilerStart()

        model_names_ctypes = [x.encode('utf-8') for x in model_names]
        lib_names = [self._model_lib[x].encode('utf-8') for x in model_names]

        # convert
        IntAr = ctypes.c_int * self._num_clients
        tids_ar = IntAr(*tids)

        CharAr = ctypes.c_char_p * self._num_clients
        model_names_ctypes_ar = CharAr(*model_names_ctypes)
        lib_names_ar = CharAr(*lib_names)

        self._sched_lib.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_char_p), POINTER(c_char_p)]

        print(model_names, lib_names, tids)

        self._sched_lib.setup(self._scheduler, self._num_clients, tids_ar, model_names_ctypes_ar, lib_names_ar)

        barrier.wait()

        num_clients = len(tids)
        print(f"Num clients is {num_clients}")
        self._sched_lib.sched_func(self._scheduler, num_clients, True)

        torch.cuda.synchronize()
