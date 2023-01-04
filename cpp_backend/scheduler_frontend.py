from ctypes import *
import torch

class PyScheduler:
    
    def __init__(self, sched_lib):
        
        torch.cuda.set_device(0)
        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib
    
    def run_scheduler(self, barrier): #queue0, mutex0):

        barrier.wait()
        #print("queue0 is: ", queue0)
        
        torch.cuda.profiler.cudart().cudaProfilerStart()
        self._sched_lib.sched_func(self._scheduler) #queue0, mutex0)

        torch.cuda.synchronize()

        torch.cuda.profiler.cudart().cudaProfilerStop()

        barrier.wait()


