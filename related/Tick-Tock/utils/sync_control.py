from utils.sync_info import SyncInfo
import torch

# These two classes make use of the `with` pattern in Python
# to centralize tick-tock synchronization logic

class ForwardControl:

    def __init__(self, thread_id: int, sync_info: SyncInfo, stream: torch.cuda.Stream) -> None:
        # we assume thread 0 starts first
        if thread_id not in {0, 1}:
            raise ValueError("thread_id can be either zero or one")

        self.sync_info = sync_info
        self.thread_id = thread_id
        self.stream = stream

    def __enter__(self) -> None:
        if self.sync_info.no_sync_control:
            return

        if self.thread_id == 0:
            self.sync_info.eventf1.wait()
            self.sync_info.eventf1.clear()
        else:
            self.sync_info.eventf0.wait()
            self.sync_info.eventf0.clear()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # wait for all the submitted kernels in the stream to complete
        self.stream.synchronize()
        if self.thread_id == 0:
            self.sync_info.eventf0.set()
        else:
            self.sync_info.eventf1.set()
        # raise the exception as is if there is any
        return exc_type is None


class BackwardControl:

    def __init__(self, thread_id: int, sync_info: SyncInfo, stream: torch.cuda.Stream) -> None:
        # we assume thread 0 starts first
        if thread_id not in {0, 1}:
            raise ValueError("thread_id can be either zero or one")

        self.sync_info = sync_info
        self.thread_id = thread_id
        self.stream = stream

    def __enter__(self) -> None:
        if self.sync_info.no_sync_control:
            return

        if self.thread_id == 0:
            self.sync_info.eventb1.wait()
            self.sync_info.eventb1.clear()
        else:
            self.sync_info.eventb0.wait()
            self.sync_info.eventb0.clear()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # wait for all the submitted kernels in the stream to complete
        self.stream.synchronize()
        if self.thread_id == 0:
            self.sync_info.eventb0.set()
        else:
            self.sync_info.eventb1.set()

        # raise the exception as is if there is any
        return exc_type is None
