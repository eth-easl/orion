import logging
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


percentile_positions = [50, 90, 95, 99]
def measure(func, num_requests, num_warm_up_reqs, request_rate, tid, shared_config, stream, sync_info: BasicSyncInfo):
    """
    Invoke the func {num_requests} times with first {num_warm_up_reqs} iterations as warm up.
    Measure how long each invocation takes and calculate statistics (average and percentiles) over them,
    and finally write all data via {sync_info}.
    """
    distribution = shared_config['distribution']
    if distribution=='trace' and tid==1:
        # uniform distribution for tid 1
        distribution = 'uniform'

    if request_rate == 0:
        intervals = [0] * num_requests
    else:
        scale = 1 / request_rate
        if distribution == 'trace':
            with open(shared_config['trace_path']) as f:
                intervals = json.load(f)
            num_requests = len(intervals)
        elif distribution == 'poisson':
            intervals = random.exponential(scale=scale, size=(num_requests,))
        elif distribution == 'uniform':
            intervals = [scale] * num_requests
        else:
            raise NotImplementedError(f'unsupported distribution {distribution}')


    latency_history = []

    with torch.no_grad():
        next_startup = time.time()
        iteration = 0
        while True:
            if time.time() >= next_startup:
                if iteration == num_warm_up_reqs:
                    sync_info.pre_measurement_prep(tid)
                    entire_inference_start_time = time.time()
                    # reset next_startup to have clear setup
                    next_startup = entire_inference_start_time

                with torch.cuda.stream(stream):
                    func()
                stream.synchronize()
                latency_history.append(1000 * (time.time() - next_startup))

                if not sync_info.should_continue_loop(tid, iteration, num_requests):
                    break

                next_startup += intervals[iteration]

                duration = next_startup - time.time()

                if duration > 0:
                    time.sleep(duration)
                iteration += 1

    inference_duration = time.time() - entire_inference_start_time
    sync_info.post_measurement_prep(tid)
    # discard the first {num_warm_up_reqs} latencies
    latency_history = latency_history[num_warm_up_reqs:]
    mean_latency = mean(latency_history)
    percentiles = np.percentile(latency_history, percentile_positions)

    # data_to_record = {
    #     f'latencies{tid}': latency_history,
    #     f'mean_latency{tid}': mean_latency,
    #     f'duration{tid}': inference_duration,
    #     f'iterations{tid}': iteration + 1,
    # }
    # record percentiles
    data_to_record = {}
    for idx, percentile_pos in enumerate(percentile_positions):
        data_to_record[f'p{percentile_pos}-latency-{tid}'] = percentiles[idx]
        data_to_record[f'throughput-{tid}'] = (iteration-num_warm_up_reqs)/inference_duration
    # write all data to the data file
    sync_info.write_kvs(data_to_record)



def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
