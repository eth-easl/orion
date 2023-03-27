import ctypes
from ctypes import *
import torch
import numpy as np
import os
import time

class PyScheduler:

    def __init__(self, sched_lib, num_clients):

        torch.cuda.set_device(0)
        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib
        self._num_clients = num_clients

    def run_scheduler(self, barriers, tids, model_names, kernel_files, num_kernels, iters):

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

        num_clients = len(tids)
        print(f"Num clients is {num_clients}")

        self._sched_lib.sched_setup(self._scheduler, num_clients, True)

        print("before starting")
        timings=[]
        torch.cuda.profiler.cudart().cudaProfilerStart()

        for i in range(iters):
            start = time.time()
            barriers[0].wait()
            #self._sched_lib.reset(num_clients)
            #barriers[0].wait()
            self._sched_lib.schedule(self._scheduler, num_clients, True)
            #self._sched_lib.schedule_one(self._scheduler, 0)
            torch.cuda.synchronize()
            #barriers[0].wait()
            # for j in range(num_clients):
            #     barriers[j].wait()
            #     self._sched_lib.schedule_one(self._scheduler, j)
            #     torch.cuda.synchronize()
            total_time = time.time()-start
            print(f"Iteration {i} took {total_time} sec")
            timings.append(total_time)
        torch.cuda.profiler.cudart().cudaProfilerStop()
        timings = timings[2:]
        print(f"Avg is {np.median(np.asarray(timings))} sec")
