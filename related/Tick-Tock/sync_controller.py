from sync_info import SyncInfo


# These two handlers make use of the `with` pattern in Python
# to encapsulate synchronization handling

class ForwardController:

    def __init__(self, thread_id: int, sync_info: SyncInfo) -> None:
        # we assume thread 0 starts first
        if thread_id not in {0, 1}:
            raise ValueError("thread_id can be either zero or one")

        self.sync_info = sync_info
        self.thread_id = thread_id

    def __enter__(self) -> None:
        if self.sync_info.no_sync_control:
            return

        if self.thread_id == 0:
            self.sync_info.eventf1.wait()
            self.sync_info.eventf1.clear()
        else:
            self.sync_info.eventf0.wait()
            self.sync_info.eventf0.clear()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.sync_info.no_sync_control:
            return

        # TODO: properly handle the exception if any
        if self.thread_id == 0:
            self.sync_info.eventf0.set()
        else:
            self.sync_info.eventf1.set()


class BackwardController:

    def __init__(self, thread_id: int, sync_info: SyncInfo) -> None:
        # we assume thread 0 starts first
        if thread_id not in {0, 1}:
            raise ValueError("thread_id can be either zero or one")

        self.sync_info = sync_info
        self.thread_id = thread_id

    def __enter__(self) -> None:
        if self.sync_info.no_sync_control:
            return

        if self.thread_id == 0:
            self.sync_info.eventb1.wait()
            self.sync_info.eventb1.clear()
        else:
            self.sync_info.eventb0.wait()
            self.sync_info.eventb0.clear()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.sync_info.no_sync_control:
            return

        # TODO: properly handle the exception if any
        if self.thread_id == 0:
            self.sync_info.eventb0.set()
        else:
            self.sync_info.eventb1.set()
