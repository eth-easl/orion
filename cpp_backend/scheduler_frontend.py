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

    def run_scheduler(self, barrier, tids, model_names, kernel_files, num_kernels):

        torch.cuda.profiler.cudart().cudaProfilerStart()

        model_names_ctypes = [x.encode('utf-8') for x in model_names]
        lib_names = [x.encode('utf-8') for x in kernel_files]

        # convert
        IntAr = ctypes.c_int * self._num_clients
        tids_ar = IntAr(*tids)
        num_kernels_ar = IntAr(*num_kernels)

        CharAr = ctypes.c_char_p * self._num_clients
        model_names_ctypes_ar = CharAr(*model_names_ctypes)
        lib_names_ar = CharAr(*lib_names)

        self._sched_lib.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_char_p), POINTER(c_char_p), POINTER(c_int)]

        print(model_names, lib_names, tids)

        self._sched_lib.setup(self._scheduler, self._num_clients, tids_ar, model_names_ctypes_ar, lib_names_ar, num_kernels_ar)

        barrier.wait()

        num_clients = len(tids)
        print(f"Num clients is {num_clients}")
        self._sched_lib.sched_func(self._scheduler, num_clients, True)

        torch.cuda.synchronize()
