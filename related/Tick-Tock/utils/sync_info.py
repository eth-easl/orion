import threading
from threading import Event


class SyncInfo:

    def __init__(
            self, eventf0: Event, eventb0: Event, eventf1: Event, eventb1: Event,  barrier: threading.Barrier,
            no_sync_control: bool = False
    ) -> None:
        # thread events - for thread synchronization
        self.eventf0 = eventf0
        self.eventf1 = eventf1
        self.eventb0 = eventb0
        self.eventb1 = eventb1
        self.barrier = barrier
        self.no_sync_control = no_sync_control
