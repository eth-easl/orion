import threading
from threading import Event
import torch


class SyncInfo:

    def __init__(
            self, eventf0: Event, eventb0: Event, eventf1: Event, eventb1: Event,
            event_cudaf0: torch.cuda.Event, event_cudab0: torch.cuda.Event, event_cudaf1: torch.cuda.Event, event_cudab1: torch.cuda.Event,
            barrier: threading.Barrier,
            no_sync_control: bool = False
    ) -> None:
        # thread events - for thread synchronization
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
