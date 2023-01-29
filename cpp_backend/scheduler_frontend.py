from ctypes import *
import torch
import numpy as np

class PyScheduler:
    
    def __init__(self, sched_lib):
        
        torch.cuda.set_device(0)
        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib
    
    def run_scheduler(self, barrier, tids): #queue0, mutex0):
        
        torch.cuda.profiler.cudart().cudaProfilerStart()

        #self._sched_lib.sched_func.argtypes = (c_void_p, c_void_p)
        self._sched_lib.setup(self._scheduler, tids[0], tids[1])

        barrier.wait()
        self._sched_lib.sched_func(self._scheduler)

        torch.cuda.synchronize()

