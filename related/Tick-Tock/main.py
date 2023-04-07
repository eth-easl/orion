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
from utils import notifier
from vision.train_imagenet import train_wrapper as vision_train_wrapper, eval_wrapper as vision_eval_wrapper
from nasnet.train_nasnet import train_wrapper as nasnet_train_wrapper
from dcgan.train_dcgan import train_wrapper as dcgan_train_wrapper
from gnmt.train_gnmt import train_wrapper as gnmt_train_wrapper
from bert.train_bert_on_squad import train_wrapper as bert_train_wrapper
from transformer.train_transformer import train_wrapper as transformer_train_wrapper
from retinanet.train_retinanet import train_wrapper as retinanet_train_wrapper

model_to_wrapper = {
    'nasnet': {
        'train': nasnet_train_wrapper,
        'eval': None,
    },
    'vision': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'dcgan': {
        'train': dcgan_train_wrapper,
        'eval': None,
    },
    'gnmt': {
        'train': gnmt_train_wrapper,
        'eval': None,
    },
    'bert': {
        'train': bert_train_wrapper,
        'eval': None,
    },
    'transformer': {
        'train': transformer_train_wrapper,
        'eval': None,
    },
    'retinanet': {
        'train': retinanet_train_wrapper,
        'eval': None,
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yaml', help='Path to the yaml config file', type=str)
parser.add_argument('--log', help='Path to the log file', type=str)

def readable_model_name(model_name, model_config):
    if model_name == 'vision':
        readable_name = f"{model_config['arc']}-{model_config['batch_size']}"
    elif model_name == 'bert':
        readable_name = f"bert-{model_config['arch']}-{model_config['batch_size']}"
    else:
        readable_name = f"{model_name}-{model_config['batch_size']}"

    return readable_name


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model0_name = config['model0']['name']
    model1_name = config['model1']['name']
    model0_mode = config['model0']['mode']
    model1_mode = config['model1']['mode']
    model0_config = config[model0_name]
    model1_config = config[model1_name]
    readable_model0_name = readable_model_name(model0_name, model0_config)
    readable_model1_name = readable_model_name(model1_name, model1_config)
    if model0_name == 'vision1':
        model0_name = 'vision'
    if model1_name == 'vision1':
        model1_name = 'vision'
    policy = config['policy']

    if args.log is None:
        args.log = f'{model0_mode}-{readable_model0_name}-{model1_mode}-{readable_model1_name}-{policy}.log'

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
    logging.info(f'full config:\n{utils.dict2pretty_str(config)}')

    logging.info(f'{model0_mode} {model0_name} and {model1_mode} {model1_name} using {policy}')
    shared_config = config['shared_config']

    data_manager = DataManager(f'{args.log}.json')

    if policy == 'MPS-process':
        sync_info = MPSSyncInfo(
            data_manager=data_manager,
            isolation_level='process'
        )
    elif policy == 'tick-tock':
        sync_info = TickTockSyncInfo(
            data_manager=data_manager
        )
    elif policy == 'MPS-thread':
        sync_info = MPSSyncInfo(
            data_manager=data_manager,
            isolation_level='thread'
        )
    elif policy == 'temporal':
        sync_info = BasicSyncInfo(data_manager, no_sync_control=True)
    else:
        raise NotImplementedError(f"unsupported policy {policy}")

    model0_wrapper = model_to_wrapper[model0_name][model0_mode]
    model1_wrapper = model_to_wrapper[model1_name][model1_mode]

    model0_kwargs = {
        'sync_info': sync_info,
        'tid': 0,
        'model_config': model0_config,
        'shared_config': shared_config
    }
    model1_kwargs = {
        'sync_info': sync_info,
        'tid': 1,
        'model_config': model1_config,
        'shared_config': shared_config
    }

    if policy == "MPS-process":
        process0 = multiprocessing.Process(target=model0_wrapper, kwargs=model0_kwargs)
        process1 = multiprocessing.Process(target=model1_wrapper, kwargs=model1_kwargs)
        process0.start()
        process1.start()

        process0.join()
        process1.join()
    elif policy == "MPS-thread":
        thread0 = threading.Thread(target=model0_wrapper, kwargs=model0_kwargs)
        thread1 = threading.Thread(target=model1_wrapper, kwargs=model1_kwargs)
        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()
    elif policy == "tick-tock":
        thread0 = threading.Thread(target=model0_wrapper, kwargs=model0_kwargs)
        thread1 = threading.Thread(target=model1_wrapper, kwargs=model1_kwargs)
        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()
    elif policy == "temporal":
        model0_wrapper(**model0_kwargs)
        model1_wrapper(**model1_kwargs)
    else:
        raise NotImplementedError(f'unsupported policy {policy}')
    # if shared_config['use_dummy_data'].enable_profiling:
    #     torch.cuda.cudart().cudaProfilerStop()

    # post-processing: sum two durations
    if policy == 'temporal':
        dict_data = data_manager.read_dict()
        duration = dict_data['duration0'] + dict_data['duration1']
        data_manager.write_kv('duration', duration)

    notifier.notify(
        subject=f'The experiment training {readable_model0_name} and {readable_model1_name} with {policy} is finished!',
        body=utils.dict2pretty_str(config)
    )
