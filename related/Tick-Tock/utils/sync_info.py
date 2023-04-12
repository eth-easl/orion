import threading
import torch
import multiprocessing
import time
from utils.data_manager import DataManager

class BasicSyncInfo:
    def __init__(self, data_manager: DataManager, no_sync_control: bool):
        self.no_sync_control = no_sync_control
        self.data_manager = data_manager

    def pre_measurement_prep(self, tid):
        return

    def post_measurement_prep(self, tid):
        return

    def write_kv(self, key, value):
        self.data_manager.write_kv(key, value)

    def write_kvs(self, kv_pairs):
        self.data_manager.write_kvs(kv_pairs)


class TickTockSyncInfo(BasicSyncInfo):

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__(data_manager, no_sync_control=False)
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        # thread events - for thread synchronization
        eventf0 = threading.Event()
        eventb0 = threading.Event()

        eventf1 = threading.Event()
        eventb1 = threading.Event()

        event_cudaf0 = torch.cuda.Event()
        event_cudab0 = torch.cuda.Event()

        event_cudaf1 = torch.cuda.Event()
        event_cudab1 = torch.cuda.Event()

        eventf1.set()  # t0 starts
        eventb1.set()

        self.eventf0 = eventf0
        self.eventf1 = eventf1
        self.eventb0 = eventb0
        self.eventb1 = eventb1
        self.event_cudaf0 = event_cudaf0
        self.event_cudab0 = event_cudab0
        self.event_cudaf1 = event_cudaf1
        self.event_cudab1 = event_cudab1
        self.start_time = None

    def pre_measurement_prep(self, tid):
        self.barrier.wait()

        if tid == 0:
            self.start_time = time.time()

    def post_measurement_prep(self, tid):
        self.no_sync_control = True
        self.barrier.wait()
        if tid == 0:
            duration = time.time() - self.start_time
            self.write_kv('duration', duration)

    def write_kv(self, key, value):
        with self.lock:
            super().write_kv(key, value)

    def write_kvs(self, kv_pairs):
        with self.lock:
            super().write_kvs(kv_pairs)

class MPSSyncInfo(BasicSyncInfo):
    def __init__(self, data_manager: DataManager, isolation_level):
        super().__init__(data_manager, no_sync_control=True)
        assert isolation_level in ['thread', 'process']
        if isolation_level == 'thread':
            self.barrier = threading.Barrier(2)
            self.lock = threading.Lock()
        else:
            self.barrier = multiprocessing.Barrier(2)
            self.lock = multiprocessing.Lock()
        self.start_time = None

    def pre_measurement_prep(self, tid):
        self.barrier.wait()
        if tid == 0:
            self.start_time = time.time()

    def post_measurement_prep(self, tid):
        self.barrier.wait()
        if tid == 0:
            duration = time.time() - self.start_time
            self.write_kv("duration", duration)

    def write_kv(self, key, value):
        with self.lock:
            super().write_kv(key, value)

    def write_kvs(self, kv_pairs):
        with self.lock:
            super().write_kvs(kv_pairs)



