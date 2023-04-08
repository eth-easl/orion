from datetime import datetime
import json
import itertools
import torch
from numpy import random
import numpy as np
import time
from statistics import mean
from utils.sync_info import BasicSyncInfo
def pretty_time():
    return datetime.now().strftime('%d-%m-%Y-%H-%M-%S')


def dict2pretty_str(dict_data):
    return json.dumps(dict_data, indent=4)


class DummyDataLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return itertools.repeat(self.batch)


def measure(func, num_requests, num_warm_up_reqs, tid, shared_config, stream, sync_info: BasicSyncInfo):
    """
    Invoke the func {num_requests} times with first {num_warm_up_reqs} iterations as warm up.
    Measure how long each invocation takes and calculate statistics (average and percentiles) over them,
    and finally write all data via {sync_info}. 
    """
    request_rate = shared_config['request_rate']
    scale = 1 / request_rate
    seed = int(time.time())
    np.random.seed(seed)
    intervals = random.exponential(scale=scale, size=(num_requests, ))
    percentile_positions = shared_config['percentile_positions']
    latency_history = []

    with torch.no_grad():
        for iter in range(num_requests):
            if iter == num_warm_up_reqs:
                # start measurement
                sync_info.pre_measurement_prep(tid)
                entire_inference_start_time = time.time()

            time.sleep(intervals[iter])
            # start func invocation
            start_time = time.time()
            with torch.cuda.stream(stream):
                func()
            stream.synchronize()
            latency = time.time() - start_time
            latency_history.append(latency)

    inference_duration = time.time() - entire_inference_start_time
    sync_info.post_measurement_prep(tid)
    # discard the first {num_warm_up_reqs} latencies
    latency_history = latency_history[num_warm_up_reqs:]
    mean_latency = mean(latency_history)
    percentiles = np.percentile(latency_history, percentile_positions)

    data_to_record = {
        f'latencies{tid}': latency_history,
        f'mean_latency{tid}': mean_latency,
        f'duration{tid}': inference_duration,
    }
    # record percentiles
    for idx, percentile_pos in enumerate(percentile_positions):
        data_to_record[f'p{percentile_pos}-{tid}'] = percentiles[idx]
    # write all data to the data file
    sync_info.write_kvs(data_to_record)


