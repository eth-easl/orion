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


    def _set(self, src, dst):
        if isinstance(src, torch.Tensor):
            dst.set_(src)
        elif isinstance(src, tuple):
            for x,y in zip(src, dst):
                self._set(x,y)



    def execute(self, s=ds):
        #print(f"op is: {self._op}, input is {self.args}")
        with torch.no_grad():
            output = self._op(*(self.args), **(self.kwargs))
        if output is not None:
            self._set(output, self._ret)
        #print(output.data_ptr(), self._ret.data_ptr())
        
        #print(f"op finished, output is {output}")
        
        return

class Scheduler:

    def __init__(self, num_clients, device, barrier, offset: Optional[int] = 0, policy: Optional[str] = None):
        self._num_clients = num_clients
        self._device = device
        self._policy = policy
        self._barrier = barrier
        self._offset = offset

        self._op_queues = [queue.Queue() for _ in range(self._num_clients)]
        
        self._lp_streams = [torch.cuda.Stream(self._device, priority=0) for _ in range(self._num_clients)]
        self._hp_stream = torch.cuda.Stream(self._device, priority=-1)

        self._executed_calls = [0 for _ in range(self._num_clients)]
        self._max_calls = []
        self._layers_info = []


        # events to capture execution of each stream, and trigger synchronized execution, one for each stream
        self._hp_event = torch.cuda.Event()
        self._lp_events = [torch.cuda.Event() for _ in range(self._num_clients)]
        
        self._stream_event_map = {}
        self._stream_map = {}
        self._stream_event_map[-1] = self._hp_event
        self._stream_map[-1] = self._hp_stream

        for i in range(self._num_clients):
            self._stream_event_map[i] =  self._lp_events[i]
            self._stream_map[i] = self._lp_streams[i]

        ## gpu-specifics
        self._max_blocks = 2560
        self._max_sms = 80

        ## for benchmarking
        self._timings = []

        ## which 'thread' runs in hp - and which prof (0: mem, 1: comp)
        self._hprun = -1

        ## to keep track of what is currently running
        self._running = set()
        self._prev_running = set()

        self._stream_prof = {}
        self._prev_stream_prof = {}

        self._dur = 0
        self._dur_pairs = 0

    def populate_info(self, prof_file, max_calls):
        # each entry is a tuple: (num_blocks, comp/mem, mem footprint)
        # for now, just use zeros
        df = pd.read_csv(prof_file)
        self._layers_info.append(list(zip(df['Num_blocks'], df['Prof'], df['Mem_footprint'], df['SM'], df['Duration(ns)'])))
        self._max_calls.append(max_calls)
        print(self._layers_info)
        print(len(self._layers_info))


    def register(self, tid, prof_file, max_calls):
        self.populate_info(prof_file, max_calls)
        return SchedClient(self._op_queues[tid])

    
    def prepare_inputs_and_run(self, operator, index, s=None):
        
        if s is None:
            torch.cuda.set_stream(self._lp_streams[0])
            operator.execute(self._lp_streams[0])
            torch.cuda.set_stream(ds) 
            self._executed_calls[index] += 1
            #layer_nr = self._executed_calls[index]
            #print(layer_nr)

        else:
            torch.cuda.set_stream(s)
            operator.execute()
            torch.cuda.set_stream(ds)
            #layer_nr = self._executed_calls[index]
            #print(layer_nr)
            
            self._executed_calls[index] += 1
        

    def schedule_seq(self):
        
        self._barrier.wait()
        start = time.time()

        total = 20
        cur = 0

        while cur < total:
    
            for i in range(self._num_clients):
                for _ in range(self._max_calls[i]):
                    next_operator = self._op_queues[i].get()
                    self.prepare_inputs_and_run(next_operator, i)

            
            torch.cuda.synchronize()

            if cur == 19:
                torch.cuda.profiler.cudart().cudaProfilerStop()


            time_iter = time.time()-start
            print(f"Scheduler loop took {time_iter*1000} ms")
            self._timings.append(time_iter*1000)
            print(f"Current median is {np.median(self._timings)} ms")

            self._barrier.wait()
            self._executed_calls = [0 for _ in range(self._num_clients)]
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
            self._queue_idx = (self._queue_idx + 1) % self._num_clients

            executed = 0
            for i in range(len(self._executed_calls)):
                if (self._executed_calls[i] == self._max_calls[i]):
                    executed += 1
            
            if executed == self._num_clients:
                # make sure everything has finished
                torch.cuda.synchronize()
                    
                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStop()


                time_iter = time.time()-start
                print(f"Scheduler loop took {time_iter*1000} ms")
                self._timings.append(time_iter*1000)
                print(f"Current median is {np.median(self._timings)} ms")
                
                self._barrier.wait()
                self._executed_calls = [0 for _ in range(self._num_clients)]
                #next_operator = None
                #return

                cur += 1

                if cur == total:
                    return


                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStart()


                self._barrier.wait()
                start = time.time()


    
    def check_waiting(self):

        if self._running == self._prev_running:
            return
        
        # new need to wait
        for new_s in self._running:
            for prev_s in self._prev_running:
                if new_s != prev_s:
                    self._stream_event_map[prev_s].wait(self._stream_map[new_s])

    def coschedule(self, op0, pr0, sm0, cnt0, prof0, id0, op1, pr1, sm1, cnt1, prof1, id1):
        
        # wait here, to avoid deadlocks
        
        self._lp_stream0 = self._lp_streams[id0]
        self._lp_stream1 = self._lp_streams[id1]
        self._lp0_event = self._lp_events[id0]
        self._lp1_event = self._lp_events[id1]


        if pr0 == 0: # run op0 with lp0
            event0 = self._lp0_event
            self._running.add(0)
            self._stream_prof[0] = prof0
        else: # run op0 with hp
            event0 = self._hp_event
            self._running.add(-1)
            self._stream_prof[-1] = prof0

        if pr1 == 0:
            event1 = self._lp1_event
            self._running.add(1)
            self._stream_prof[1] = prof1
        else:
            event1 = self._hp_event
            self._running.add(-1)
            self._stream_prof[-1] = prof1
        
        if pr0 == 1:
            self._hprun = id0
        elif pr1 == 1:
            self._hprun = id1
        else:
            self._hprun = -1
        
        self.check_waiting()

        self._cos += 1

        if pr0 == pr1:
            # both are low priority
            self.prepare_inputs_and_run(op0, id0, self._lp_stream0)
            event0.record(self._lp_stream0)
            self.prepare_inputs_and_run(op1, id1, self._lp_stream1)
            event1.record(self._lp_stream1)

        elif pr0 > pr1:
        
            self.prepare_inputs_and_run(op1, id1, self._lp_stream1)
            event1.record(self._lp_stream1)

            self.prepare_inputs_and_run(op0, id0, self._hp_stream)
            event0.record(self._hp_stream)

        
        elif pr1 > pr0:

            self.prepare_inputs_and_run(op0, id0, self._lp_stream0)
            event0.record(self._lp_stream0)

            self.prepare_inputs_and_run(op1, id1, self._hp_stream)
            event1.record(self._hp_stream)


    def schedule_one(self, index, op, prof, cnt):

        #print("------ schedule one!!!!!!!!!!!!!")

        self._dur += self._layers_info[index][cnt][-1]
        stream = self._lp_streams[index]
        event =  self._lp_events[index]

        self._running.add(index)
        self._stream_prof[index] = prof

        self.check_waiting()

        self.prepare_inputs_and_run(op, index, stream)
        event.record(stream)


    def schedule_blocks(self, op0, cnt0, id0, op1, cnt1, id1):
        
        start = time.time()
        b0, p0, m0, sm0, d0 = self._layers_info[id0][cnt0]
        b1, p1, m1, sm1, d1 = self._layers_info[id1][cnt1]
        ret_op0 = None
        ret_op1 = None
    

        if p0 > -1 and p0==p1:
            self.schedule_one(id0, op0, p0, cnt0)
            return None, op1

        # different profiles
        if sm0 < self._max_sms and  sm1 < self._max_sms:
            self._dur_pairs += max(d0, d1)
            self.coschedule(op0, 0, sm0, cnt0, p0, id0, op1, 0, sm1, cnt1, p1, id1)

        elif sm0 >= self._max_sms and sm1 < self._max_sms:
            self._dur_pairs += max(d0, d1)
            self.coschedule(op0, 0, sm0, cnt0, p0, id0, op1, 1, sm1, cnt1, p1, id1)

        elif sm0 < self._max_sms and sm1 >= self._max_sms:
            self._dur_pairs += max(d0, d1)
            self.coschedule(op0, 1, sm0, cnt0, p0, id0, op1, 0, sm1, cnt1, p1, id1)
        
        else:
            # TODO: not sure what to do at this case
            # current policy: when they don't fit in any way, schedule them sequentially
            self.schedule_one(id0, op0, p0, cnt0)
            ret_op1 = op1

        return ret_op0, ret_op1
    
    def schedule_profile(self):
 
        it = 0
        next_op0 = None
        next_op1 = None
            
        iters = 0

        self._barrier.wait()
        start = time.time()

        total = 20
        cur = 0

        self._cos = 0

        client0 = 0
        client1 = 1

        while True:
        
            start_s = time.time()
            
            if next_op0 is None:
                next_op0 = self._op_queues[client0].get()

            if next_op1 is None and client1 > 0 and it >= self._offset:
                next_op1 = self._op_queues[client1].get()

            it += 1
            
            #print(self._executed_calls)

            self._running = set()
            self._stream_prof = {}

            if next_op0 is not None and next_op1 is not None:
                next_op0, next_op1 = self.schedule_blocks(next_op0, self._executed_calls[client0], client0, next_op1, self._executed_calls[client1], client1)
           
            elif next_op0 is not None:
                self.schedule_one(client0, next_op0, self._layers_info[0][self._executed_calls[client0]][1], self._executed_calls[client0])
                next_op0 = None

            elif next_op1 is not None:
                self.schedule_one(client1, next_op1, self._layers_info[1][self._executed_calls[client1]][1], self._executed_calls[client1])
                next_op1 = None

            if self._executed_calls[client0] == self._max_calls[client0]:
                client0 = client1
                next_op0 = next_op1
                client1 += 1
                next_op1 = None
                if client1 >= self._num_clients:
                    client1 = -1
 

            #print(self._running)
            self._prev_running = self._running
            self._prev_stream_prof = self._stream_prof

            executed = 0
            for i in range(len(self._executed_calls)):
                if (self._executed_calls[i] == self._max_calls[i]):
                    executed += 1

            if executed == self._num_clients:
                # make sure everything has finished

                torch.cuda.synchronize()

                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStop()
                    #torch.cuda.nvtx.range_pop()
                
                time_iter = time.time()-start
                
                dur_ms = self._dur / 1000000
                print(f"Scheduler loop took {time_iter*1000} ms, it is {it}, {self._cos} coscheduled, indiv time is {dur_ms} ms, pair time is {self._dur_pairs/1000000} ms")

                self._cos = 0

                self._timings.append(time_iter*1000)
                #print(f"Current median is {np.median(self._timings)} ms")
                # 'notify' the trainers that operations have completed
                self._barrier.wait()
                self._executed_calls = [0 for _ in range(self._num_clients)]
                self._dur = 0
                self._dur_pairs = 0

                it = 0
                cur += 1

                if cur == total:
                    self._timings = self._timings[2:]
                    print(self._timings)
                    print(f"Median is {np.median(self._timings)} ms")
                    return


                if cur == 19:
                    torch.cuda.profiler.cudart().cudaProfilerStart() 
                    #print("start")
                    #torch.cuda.nvtx.range_push("start")
                
                client0 = 0
                client1 = 1
                next_op0 = None
                next_op1 = None

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
