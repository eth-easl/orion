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
    if request_rate > 0:
        scale = 1 / request_rate
        intervals = random.exponential(scale=scale, size=(num_requests,))
    else:
        intervals = [0]*num_requests
    seed = int(time.time())
    np.random.seed(seed)
    percentile_positions = shared_config['percentile_positions']
    latency_history = []

    with torch.no_grad():
        for iteration in range(num_requests):
            if iteration == num_warm_up_reqs:
                # start measurement
                sync_info.pre_measurement_prep(tid)
                entire_inference_start_time = time.time()

            time.sleep(intervals[iteration])
            # start func invocation
            with torch.cuda.stream(stream):
                start_time = time.time()
                func()
            stream.synchronize()
            latency = time.time() - start_time
            # convert to ms
            latency_history.append(latency * 1000)

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


def non_stop_measure(func, num_warm_up_reqs, tid, shared_config, stream, sync_info: BasicSyncInfo):
    """
    Invoke the func continuously with first {num_warm_up_reqs} iterations as warm up until {sync_info} instructs
    it to stop.
    Measure how long each invocation takes and calculate statistics (average and percentiles) over them,
    and finally write all data via {sync_info}.
    """
    request_rate = shared_config['request_rate']

    seed = int(time.time())
    np.random.seed(seed)
    percentile_positions = shared_config['percentile_positions']
    latency_history = []

    iteration = -1
    with torch.no_grad():
        while sync_info.should_continue_loop():
            iteration += 1
            if iteration == num_warm_up_reqs:
                # start measurement
                sync_info.pre_measurement_prep(tid)
                entire_inference_start_time = time.time()

            if request_rate > 0:
                time.sleep(random.exponential(scale=1/request_rate))

            # start func invocation
            with torch.cuda.stream(stream):
                start_time = time.time()
                func()
            stream.synchronize()
            latency = time.time() - start_time
            # convert to ms
            latency_history.append(latency * 1000)

    inference_duration = time.time() - entire_inference_start_time
    sync_info.post_measurement_prep(tid)
    # discard the first {num_warm_up_reqs} latencies and the last latency as part if it was being processed when
    # the other thread/process had finished
    latency_history = latency_history[num_warm_up_reqs:-1]
    mean_latency = mean(latency_history)
    percentiles = np.percentile(latency_history, percentile_positions)

    data_to_record = {
        f'latencies{tid}': latency_history,
        f'mean_latency{tid}': mean_latency,
        f'duration{tid}': inference_duration,
        f'iterations{tid}': iteration + 1
    }
    # record percentiles
    for idx, percentile_pos in enumerate(percentile_positions):
        data_to_record[f'p{percentile_pos}-{tid}'] = percentiles[idx]
    # write all data to the data file
    sync_info.write_kvs(data_to_record)

