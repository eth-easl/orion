import torch
import threading

class SyncInfo:
    
    def __init__(
            self,
            barrier,
            event_cudaf0,
            event_cudab0,
            eventf0,
            eventb0,
            event_cudaf1,
            event_cudab1,
            eventf1,
            eventb1,
        ) -> None:

    
        self.barrier = barrier

        # cuda events - for cuda stream synchronization
        self.event_cudaf0 = event_cudaf0
        self.event_cudaf1 = event_cudaf1
        self.event_cudab0 = event_cudab0
        self.event_cudab1 = event_cudab1

        # thread events - for thread synchronization
        self.eventf0 = eventf0
        self.eventf1 = eventf1
        self.eventb0 = eventb0
        self.eventb1 = eventb1
