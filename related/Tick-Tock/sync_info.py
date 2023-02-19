from threading import Event


class SyncInfo:

    def __init__(
            self,
            barrier,
            eventf0: Event,
            eventb0: Event,
            eventf1: Event,
            eventb1: Event,
    ) -> None:
        self.barrier = barrier

        # thread events - for thread synchronization
        # as cpu submits the kernel and synchronously waits for the response
        # cuda event is not needed
        self.eventf0 = eventf0
        self.eventf1 = eventf1
        self.eventb0 = eventb0
        self.eventb1 = eventb1
