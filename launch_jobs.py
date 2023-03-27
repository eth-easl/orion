import argparse
import json
import threading
import time

from scheduler import Scheduler
from examples.train_imagenet import imagenet_loop
from examples.draft_modules import custom_model_loop

import sys
sys.path.append("/home/fot/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
from train import run_transformer

parser = argparse.ArgumentParser(description='Launcher of ML workloads')
parser.add_argument('--info_file', type=str, help='json file containing the description of the ML jobs to be launched')
parser.add_argument('--policy', default='profile', type=str, help='scheduling policy')
parser.add_argument('--offset', type=int, default=0, help='offset of the stream execution')
parser.add_argument('--device', type=int, default=0, help='GPU to run on')


func_map = {
    'imagenet_loop': imagenet_loop,
    'run_transformer': run_transformer,
    'custom_model_loop': custom_model_loop,
}

def parse_input_file(job_file):
    job_desc = json.load(open(job_file, 'r'))
    job_desc_list = []
    for k,v in job_desc.items():
        job_desc_list.append(v)
    return job_desc_list

def launch_jobs(job_list, policy, device, offset):
    num_jobs = len(job_list)

    if policy == "profile":
        assert num_jobs >= 2

    barrier = threading.Barrier(num_jobs+1)
    layer_sched = Scheduler(num_jobs, device, policy=policy, barrier=barrier, offset=offset)

    layer_queues = []
    for i in range(num_jobs):
        layer_queues.append(layer_sched.register(i, job_list[i]['layers_file'], job_list[i]['max_calls']))


    print("start scheduling thread")
    sched_thread = threading.Thread(target=layer_sched.schedule)
    sched_thread.start()

    print("start training threads")

    # could be the same or different functions
    train_threads = []
    for i in range(num_jobs):
        train_threads.append(
                threading.Thread(target=func_map[job_list[i]['func']], args=(layer_queues[i], job_list[i]['arch'], job_list[i]['batchsize'], None, device, barrier, i))
                #threading.Thread(target=func_map[job_list[i]['func']], args=(layer_queues[i], barrier, i, device, job_list[i]['batchsize']))
        )

    start = time.time()
    for i in range(num_jobs):
        train_threads[i].start()

    for i in range(num_jobs):
        train_threads[i].join()

    sched_thread.join()

    print(f"-------------------------- all threads joined!It took: {time.time()-start}", flush=True)



if __name__ == "__main__":
    args = parser.parse_args()
    job_list = parse_input_file(args.info_file)
    print(job_list)
    launch_jobs(job_list, args.policy, args.device, args.offset)
