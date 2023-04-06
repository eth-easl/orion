import threading
import torch
import multiprocessing
import time
from utils.data_writer import DataWriter

class BasicSyncInfo:
    def __init__(self, data_writer: DataWriter, no_sync_control: bool):
        self.no_sync_control = no_sync_control
        self.data_writer = data_writer

    def pre_measurement_prep(self, tid):
        return

    def post_measurement_prep(self, tid):
        return

    def write_kv(self, key, value):
        self.data_writer.write_kv(key, value)


class TickTockSyncInfo(BasicSyncInfo):

    def __init__(self, data_writer: DataWriter) -> None:
        super().__init__(data_writer, no_sync_control=False)
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
        self.barrier.wait()
        if tid == 0:
            duration = time.time() - self.start_time
            self.write_kv('duration', duration)

    def write_kv(self, key, value):
        with self.lock:
            super().write_kv(key, value)

class MPSSyncInfo(BasicSyncInfo):
    def __init__(self, data_writer: DataWriter, isolation_level):
        super().__init__(data_writer, no_sync_control=True)
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



