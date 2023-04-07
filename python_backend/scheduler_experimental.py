from typing import Optional

import torch
import queue
import time
import pandas as pd
import numpy as np



c1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
input_c1 = torch.randn(2, 3, 224, 224).cuda()

b1 = torch.nn.BatchNorm2d(64).cuda()
input_b1 = torch.randn(64, 64, 112, 112).cuda()


### remaining questions:
## 1. what to do when they don't fit?
## 2. why defining a part of code to run with a stream is time-consuming?

### Comments:
# this version of the scheduler uses:
# 1) 'set_stream' to set the current stream
# 2) no events for synchronization across streams
# these two changes lead to faster implementation
# THIS IS EXPERIMENTAL!!!!!!!!!!!!!!!!!!!!!!!!!

ds = torch.cuda.default_stream()
s1 = torch.cuda.Stream(priority=0)
s2 = torch.cuda.Stream(priority=0)


print(f"default stream is {ds}")

class Operator:

    def __init__(self, op, profile, ret, *args, **kwargs):
        self._op = op
        self._ret = ret

        # public
        self.profile = profile
        self.args = args
        self.kwargs = kwargs


    def _copy(self,src, dst):

        if isinstance(src, torch.Tensor):
            dst.copy_(src)
        elif isinstance(src, tuple):
            for x,y in zip(src, dst):
                self._copy(x,y)

    def execute(self, s=ds):
        # TODO: is copy really needed here?
        #print(f"op is: {self._op}, input is {self.args}")
        with torch.no_grad():
            output = self._op(*(self.args), **(self.kwargs))
        self._ret.set_(output)
        #print(output.data_ptr(), self._ret.data_ptr())
        
        #print(f"op finished, output is {output}")
        
        return

class Scheduler:

    def __init__(self, prof_file, num_streams, device, max_calls, barrier, policy: Optional[str] = None):
        self._num_streams = num_streams
        self._device = device
        self._policy = policy
        self._max_calls = max_calls
        self._barrier = barrier

        self._op_queues = [queue.Queue() for _ in range(self._num_streams)]
        #self._streams = [torch.cuda.Stream(self._device) for _ in range(self._num_streams)]
        
        self._lp_stream0 = torch.cuda.Stream(self._device, priority=0)
        self._lp_stream1 = torch.cuda.Stream(self._device, priority=0)
        self._hp_stream = torch.cuda.Stream(self._device, priority=-1)

        self._executed_calls = [0 for _ in range(self._num_streams)]

        self.populate_info(prof_file)

        # events to capture execution of each stream, and trigger synchronized execution, one for each stream
        self._hp_event = torch.cuda.Event()
        self._lp0_event = torch.cuda.Event()
        self._lp1_event = torch.cuda.Event()

        ## gpu-specifics
        self._max_blocks = 2560


        ## for benchmarking
        self._timings = []

    def populate_info(self, prof_file):
        # each entry is a tuple: (num_blocks, comp/mem, mem footprint)
        # for now, just use zeros
        df = pd.read_csv(prof_file)
        self._layers_info = list(zip(df['Num_blocks'], df['Prof'], df['Mem_footprint']))
        print(self._layers_info)
        print(len(self._layers_info))


    def register(self, tid):
        return SchedClient(self._op_queues[tid])

    
    def prepare_inputs_and_run(self, operator, index, s=None):
        
        if s is None:
            torch.cuda.set_stream(self._lp_stream0)
            operator.execute(self._lp_stream0)
            torch.cuda.set_stream(ds) 
            self._executed_calls[index] += 1
            
        else:
            torch.cuda.set_stream(s)
            operator.execute()
            torch.cuda.set_stream(ds)
            #layer_nr = self._executed_calls[index]
            self._executed_calls[index] += 1
        

    def schedule_seq(self):
        
        self._barrier.wait()
        start = time.time()

        total = 50
        cur = 0

        while cur < total:
    
            # run all from queue 0
            for _ in range(self._max_calls):
                next_operator = self._op_queues[0].get()
                self.prepare_inputs_and_run(next_operator, 0)

            # if queue 1, run all from queue 1
            if self._num_streams == 2:
                for _ in range(self._max_calls):
                    next_operator = self._op_queues[1].get()
                    self.prepare_inputs_and_run(next_operator, 1)

            
            torch.cuda.synchronize()

            if cur == 19:
                torch.cuda.profiler.cudart().cudaProfilerStop()


            time_iter = time.time()-start
            print(f"Scheduler loop took {time_iter*1000} ms")
            self._timings.append(time_iter*1000)
            print(f"Current median is {np.median(self._timings)} ms")

            self._barrier.wait()
            self._executed_calls = [0 for _ in range(self._num_streams)]
            #next_operator = None
            #return

            cur += 1
            if cur == total:
                return
            
            if cur == 19:
                torch.cuda.profiler.cudart().cudaProfilerStart()

            
            self._barrier.wait()
            start = time.time()




    def schedule_rr(self):
 

        self._queue_idx = 0
        
        self._barrier.wait()
        start = time.time()
        
        next_operator = None

        total = 50
        cur = 0

        while True:

            # prepare inputs and execute
            #if next_operator is None:
            next_operator = self._op_queues[self._queue_idx].get()
            
            self.prepare_inputs_and_run(next_operator, self._queue_idx)

            self._queue_idx = (self._queue_idx + 1) % self._num_streams

            executed = [x for x in self._executed_calls if x == self._max_calls]
            
            if len(executed) == self._num_streams:
                # make sure everything has finished
                torch.cuda.synchronize()
                    
                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStop()


                time_iter = time.time()-start
                print(f"Scheduler loop took {time_iter*1000} ms")
                self._timings.append(time_iter*1000)
                print(f"Current median is {np.median(self._timings)} ms")
                
                self._barrier.wait()
                self._executed_calls = [0 for _ in range(self._num_streams)]
                #next_operator = None
                #return

                cur += 1

                if cur == total:
                    return


                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStart()


                self._barrier.wait()
                start = time.time()

    def coschedule(self, op0, pr0, b0, op1, pr1, b1):
        
        # wait here, to avoid deadlocks
        
        if pr0 == 0: # run op0 with lp0
            #self._hp_event.wait(self._lp_stream0)
            #self._lp1_event.wait(self._lp_stream0)
            event0 = self._lp0_event
        else: # run op0 with hp
            #self._lp0_event.wait(self._hp_stream)
            #self._lp1_event.wait(self._hp_stream)
            event0 = self._hp_event

        if pr1 == 0:
            #self._hp_event.wait(self._lp_stream1)
            #self._lp0_event.wait(self._lp_stream1)
            event1 = self._lp1_event
        else:
            #self._lp1_event.wait(self._hp_stream)
            #self._lp0_event.wait(self._hp_stream)
            event1 = self._hp_event
        
        #print("-------- coschedule ", op0._op, op1._op, b0, b1, pr0, pr1)
        
        self._cos += 1

        if pr0 == pr1:
            # both are low priority
            self.prepare_inputs_and_run(op0, 0, self._lp_stream0)
            #event0.record(self._lp_stream0)

            self.prepare_inputs_and_run(op1, 1, self._lp_stream1)
            #event1.record(self._lp_stream1)

        elif pr0 > pr1:
        

            self.prepare_inputs_and_run(op0, 0, self._hp_stream)
            #event0.record(self._hp_stream)

            self.prepare_inputs_and_run(op1, 1, self._lp_stream1)
            #event1.record(self._lp_stream1)
        
        elif pr1 > pr0:


            self.prepare_inputs_and_run(op1, 1, self._hp_stream)
            #event1.record(self._hp_stream)

            self.prepare_inputs_and_run(op0, 0, self._lp_stream0)
            #event0.record(self._lp_stream0)

    def schedule_one(self, index, op):

        if index == 0:
            stream = self._lp_stream0
            #self._hp_event.wait(stream)
            #self._lp1_event.wait(stream)
            event = self._lp0_event
        else:
            stream = self._lp_stream1
            #self._hp_event.wait(stream)
            #self._lp0_event.wait(stream)
            event = self._lp1_event

        self.prepare_inputs_and_run(op, index, stream)
        #event.record(stream)


    def schedule_blocks(self, op0, cnt0, op1, cnt1):
        
        start = time.time()
        b0, p0, m0 = self._layers_info[cnt0]
        b1, p1, m1 = self._layers_info[cnt1]
        

        ret_op0 = None
        ret_op1 = None

        if b0 < self._max_blocks and  b1 < self._max_blocks:
            self.coschedule(op0, 0, b0, op1, 0, b1)

        elif b0 >= self._max_blocks and b1 < self._max_blocks:
            self.coschedule(op0, 0, b0, op1, 1, b1)

        elif b0 < self._max_blocks and b1 >= self._max_blocks:
            self.coschedule(op0, 1, b0, op1, 0, b1)

        else:
            # TODO: not sure what to do at this case
            # current policy: when they don't fit in any way, schedule them sequentially
            
            #print("here")
            self.schedule_one(0, op0)
            #self.schedule_one(1, op1)
            ret_op1 = op1

        return ret_op0, ret_op1

    def schedule_profile(self):
 
        it = 0
        next_op0 = None
        next_op1 = None
            
        iters = 0

        self._barrier.wait()


        start = time.time()

        total = 50
        cur = 0

        self._cos = 0

        while True:
        
            start_s = time.time()
            
            '''
            if not self._op_queues[0].empty():
                next_op0 = self._op_queues[0].queue[0]
            else:
                next_op0 = None

            if not  self._op_queues[1].empty():
                next_op1 = self._op_queues[1].queue[0]
            else:
                next_op1 = None
            '''

            if next_op0 is None and self._executed_calls[0] < self._max_calls:
                next_op0 = self._op_queues[0].get()
            #else:
            #    next_op0 = None

            if next_op1 is None and self._executed_calls[1] < self._max_calls:
                next_op1 = self._op_queues[1].get()
            #else:
            #    next_op1 = None
            

            #print(f"--- Scheduler, Current stream is {torch.cuda.current_stream()}")


            #print(f"it: {it}, getting took {(time.time()-start_s)*1000} ms")
            it += 1
            
            
            if next_op0 is not None:
                if next_op1 is not None:
                            
                    next_op0, next_op1 = self.schedule_blocks(next_op0, self._executed_calls[0], next_op1, self._executed_calls[1])

                else:
                    #self._executed_calls[0] += 1
                    #self._op_queues[0].get()
                    self.schedule_one(0, next_op0)
                    next_op0 = None
                    
            elif next_op1 is not None:
                #self._executed_calls[1] += 1
                #self._op_queues[1].get()
                self.schedule_one(1, next_op1)
                next_op1 = None

            executed = [x for x in self._executed_calls if x == self._max_calls]
    
            if len(executed) == self._num_streams:
                # make sure everything has finished
                torch.cuda.synchronize()
                
                print(self._op_queues[0].qsize(), self._op_queues[1].qsize())

                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStop()

                time_iter = time.time()-start
                print(f"Scheduler loop took {time_iter*1000} ms, it is {it}, {self._cos} coscheduled")

                self._cos = 0
                self._timings.append(time_iter*1000)
                #print(f"Current median is {np.median(self._timings)} ms")
                # 'notify' the trainers that operations have completed
                self._barrier.wait()
                self._executed_calls = [0 for _ in range(self._num_streams)]
                it = 0
                cur += 1

                if cur == total:
                    self._timings = self._timings[2:]
                    print(sorted(self._timings))
                    print(f"Median is {np.median(self._timings)} ms")
                    return


                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStart() 

                # make sure queues are filled 
                self._barrier.wait()
                start = time.time()

    


    def schedule(self):

        self._barrier.wait()
        print("Enter scheduling loop!")
        
        
        if self._policy == "round_robin":
            self.schedule_rr()
        elif self._policy == "profile":
            self.schedule_profile()
        elif self._policy == "sequential":
            self.schedule_seq()
        else:
            print("Only Round-Robin and profile-based policy are supported for now, abort!")
            return
        

class SchedClient:

    def __init__(self, op_queue):
        self._op_queue = op_queue

    def enqueue(self, operator, profile, ret, *args, **kwargs):
        self._op_queue.put(Operator(operator, profile, ret, *args, **kwargs))

    def op_queue(self):
        return self._op_queue
