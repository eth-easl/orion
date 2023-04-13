import ctypes
from ctypes import *
import torch
import numpy as np
import os
import time

class PyScheduler:

    def __init__(self, num_clients):

        home_directory = os.path.expanduser( '~' )
        self._sched_lib = cdll.LoadLibrary(home_directory + "/gpu_share_repo/mps_backend/scheduler/scheduler.so")
        self._scheduler =  self._sched_lib.sched_init()

        self._num_clients = num_clients

    def run_scheduler(
        self,
        barriers,
        tids,
        model_names,
        kernel_files,
        additional_kernel_files,
        num_kernels,
        additional_num_kernels,
        num_iters,
        profile,
        run_eval,
        reef
    ):

        model_names_ctypes = [x.encode('utf-8') for x in model_names]
        lib_names = [x.encode('utf-8') for x in kernel_files]

        # # convert
        IntAr = ctypes.c_int * self._num_clients
        tids_ar = IntAr(*tids)
        num_kernels_ar = IntAr(*num_kernels)
        num_iters_ar = IntAr(*num_iters)

        CharAr = ctypes.c_char_p * self._num_clients
        model_names_ctypes_ar = CharAr(*model_names_ctypes)
        lib_names_ar = CharAr(*lib_names)

        self._sched_lib.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_char_p), POINTER(c_char_p), POINTER(c_int)]

        print(model_names, lib_names, tids)

        self._sched_lib.setup(self._scheduler, self._num_clients, tids_ar, model_names_ctypes_ar, lib_names_ar, num_kernels_ar, num_iters_ar)

        num_clients = len(tids)
        print(f"Num clients is {num_clients}")

        self._sched_lib.sched_setup(self._scheduler, num_clients, profile, reef)

        print(f"before starting, profile is {profile}")
        # timings=[]
        # torch.cuda.profiler.cudart().cudaProfilerStart()
        torch.cuda.synchronize()

        if run_eval:
            if profile:
                barriers[0].wait()
                # run once to warm-up and setup
                # self._sched_lib.schedule(self._scheduler, num_clients, True, 0, True, reef)
                # torch.cuda.synchronize()

                # for j in range(num_clients):
                #     if (additional_kernel_files[j] is not None):
                #         new_kernel_file = additional_kernel_files[0].encode('utf-8')
                #         self._sched_lib.setup_change(self._scheduler, j, new_kernel_file, additional_num_kernels[j])

                # print("wait here")
                # barriers[0].wait()
                # print("done!")


                start = time.time()
                print("call schedule")
                self._sched_lib.schedule(self._scheduler, num_clients, True, 0, False, reef)
                barriers[0].wait()

                torch.cuda.synchronize()
                print(f"Total time is {time.time()-start}")
