import os
import threading
from threading import Event
import torch
import time
import json

class SyncInfo:

    def __init__(
            self, barrier, experiment_data_json_file, no_sync_control: bool = False
    ) -> None:
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
        self.barrier = barrier
        self.no_sync_control = no_sync_control
        self.experiment_data_json_file = experiment_data_json_file
        self.start_time = None

    def pre_measurement_prep(self, tid):
        if not self.no_sync_control:
            self.barrier.wait()
        if tid == 0:
            self.start_time = time.time()

    def post_measurement_prep(self, tid):
        if self.no_sync_control:
            return

        self.barrier.wait()
        if tid == 0:
            duration = time.time() - self.start_time
            with open(self.experiment_data_json_file, 'w') as f:
                json.dump({'duration': duration}, f, indent=4)

class MPSSyncInfo():
    def __init__(self, experiment_data_json_file, barrier):
        # so that the tick-tock related code won't work
        self.no_sync_control = True
        self.experiment_data_json_file = experiment_data_json_file
        self.barrier = barrier
        self.start_time = None

    def pre_measurement_prep(self, tid):
        self.barrier.wait()
        if tid == 0:
            self.start_time = time.time()

    def post_measurement_prep(self, tid):
        self.barrier.wait()
        if tid == 0:
            duration = time.time() - self.start_time
            with open(self.experiment_data_json_file, 'w') as f:
                json.dump({'duration': duration}, f, indent=4)

