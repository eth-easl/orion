import torch
import threading
import multiprocessing
import logging
import yaml
import time
import argparse
import utils
from utils.sync_info import *
from utils.data_manager import DataManager
from vision.train_imagenet import train_wrapper as vision_train_wrapper, eval_wrapper as vision_eval_wrapper
from bert.train_bert_on_squad import train_wrapper as bert_train_wrapper, eval_wrapper as bert_eval_wrapper
from transformer.train_transformer import train_wrapper as transformer_train_wrapper, eval_wrapper as transformer_eval_wrapper

model_to_wrapper = {
    'resnet50': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'resnet101': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'mobilenet_v2': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'bert': {
        'train': bert_train_wrapper,
        'eval': bert_eval_wrapper,
    },
    'transformer': {
        'train': transformer_train_wrapper,
        'eval': transformer_eval_wrapper,
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yaml', help='Path to the yaml config file', type=str)
parser.add_argument('--log', help='Path to the log file', type=str)



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    models = config['models']
    model_names = []
    model_modes = []
    model_configs = []

    print(models)

    model_names = [model_dict['name'] for _,model_dict in models.items()]
    model_modes = [model_dict['mode'] for _,model_dict in models.items()]
    model_configs = [config[mname] for mname in model_names]

    print(model_names, model_modes, model_configs)
    num_clients = len(models)
    print(f"num_clients is {num_clients}")

    policy = config['policy']
    if policy=="TickTock" and num_clients != 2:
        raise ValueError("Tick-Tock scheduling policy requires exactly 2 clients!")

    if args.log is None:
        args.log = ""
        for mmode,mname in zip(model_modes, model_names):
            args.log += f"{mmode}-{mname}"
        args.log += ".log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        handlers=[
            # also output to console
            # logging.StreamHandler(),
            logging.FileHandler(args.log, mode='a'),
        ]
    )


    shared_config = config['shared_config']

    logging.info(f'full config:\n{utils.dict2pretty_str(config)}')
    data_manager = DataManager(f'{args.log}.json')

    if policy == 'MPS':
        sync_info = ConcurrentSyncInfo(
            data_manager=data_manager,
            num_clients=num_clients,
            isolation_level='process'
        )
    elif policy == 'TickTock':
        sync_info = TickTockSyncInfo(
            data_manager=data_manager
        )
    elif policy == 'Streams':
        sync_info = ConcurrentSyncInfo(
            data_manager=data_manager,
            num_clients=num_clients,
            isolation_level='thread'
        )
    elif policy == 'Isolated':
        sync_info = BasicSyncInfo(data_manager, no_sync_control=True)
    elif policy == 'Sequential':
        sync_info = ConcurrentSyncInfo(
            data_manager=data_manager,
            num_clients=num_clients,
            isolation_level='thread'
        )
        shared_config['default'] = True
    else:
        raise NotImplementedError(f"unsupported policy {policy}")

    for i in range(num_clients):
        if model_names[i][-2:] == '-1':
             model_names[i] =  model_names[i][:-2]

    model_wrappers = [model_to_wrapper[mname][mmode] for mname,mmode in zip(model_names, model_modes)]
    model_kwargs = []
    for i,mconfig in enumerate(model_configs):
        model_kwargs.append(
            {
                'sync_info': sync_info,
                'tid': i,
                'model_config': mconfig,
                'shared_config': shared_config
            }
        )

    if policy == "MPS":
        processes = [multiprocessing.Process(target=mwrapper, kwargs=mkwargs) for mwrapper,mkwargs in zip(model_wrappers, model_kwargs)]
        for i in range(num_clients):
            processes[i].start()

        for i in range(num_clients):
            processes[i].join()
    elif policy == "Isolated":
        for mwrapper, mkwargs in zip(model_wrappers, model0_kwargs):
            mwrapper(**mkwargs)
    elif policy in {"Streams", 'TickTock', 'Sequential'}:
        threads = [threading.Thread(target=mwrapper, kwargs=mkwargs) for mwrapper,mkwargs in zip(model_wrappers, model_kwargs)]
        for i in range(num_clients):
            threads[i].start()

        for i in range(num_clients):
            threads[i].join()
    else:
        raise NotImplementedError(f'unsupported policy {policy}')

    # post-processing: sum two durations
    if policy == 'Isolated':
        dict_data = data_manager.read_dict()
        duration = dict_data['duration0'] #+ dict_data['duration1']
        data_manager.write_kv('duration', duration)
