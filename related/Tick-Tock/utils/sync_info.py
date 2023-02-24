from threading import Event


class SyncInfo:

    def __init__(
            self, eventf0: Event, eventb0: Event, eventf1: Event, eventb1: Event, no_sync_control: bool = False
    ) -> None:
        # thread events - for thread synchronization
        # as cpu submits the kernel and synchronously waits for the response
        # cuda event is not needed
        self.eventf0 = eventf0
        self.eventf1 = eventf1
        self.eventb0 = eventb0
        self.eventb1 = eventb1
        self.no_sync_control = no_sync_control
