import torch
import threading
import logging
import yaml
import time
import argparse
import utils
from utils.sync_info import SyncInfo
import utils.constants as constants
from utils import notifier
from vision.train_imagenet import train_wrapper as vision_train_wrapper
from nasnet.train_nasnet import train_wrapper as nasnet_train_wrapper
from dcgan.train_dcgan import train_wrapper as dcgan_train_wrapper
from gnmt.train_gnmt import train_wrapper as gnmt_train_wrapper
from bert.train_bert_on_squad import train_wrapper as bert_train_wrapper
from transformer.train_transformer import train_wrapper as transformer_train_wrapper
from retinanet.train_retinanet import train_wrapper as retinanet_train_wrapper


model_to_train_wrapper = {
    'nasnet': nasnet_train_wrapper,
    'vision': vision_train_wrapper,
    'dcgan': dcgan_train_wrapper,
    'gnmt': gnmt_train_wrapper,
    'bert': bert_train_wrapper,
    'transformer': transformer_train_wrapper,
    'retinanet': retinanet_train_wrapper,
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yaml', help='Path to the yaml config file', type=str)
parser.add_argument('--log', default= f'./{utils.pretty_time()}-training.log', help='Path to the log file', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
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

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    logging.info(f'full config:\n{utils.dict2pretty_str(config)}')
    model0_name = config['model0']
    model1_name = config['model1']
    policy = config['policy']
    logging.info(f'start training with {model0_name} and {model1_name} using {policy}')
    for key, value in config['data_dir'].items():
        constants.__dict__[key] = value

    device = torch.device("cuda:0")

    stream0 = torch.cuda.Stream(device=device)
    stream1 = torch.cuda.Stream(device=device)

    eventf0 = threading.Event()
    eventb0 = threading.Event()

    eventf1 = threading.Event()
    eventb1 = threading.Event()

    eventf1.set()  # t0 starts
    eventb1.set()

    num_epochs = config['num_epochs']
    logging.info(f'number of epochs: {num_epochs}')
    sync_info = SyncInfo(eventf0, eventb0, eventf1, eventb1, barrier=threading.Barrier(2))

    model0_train_wrapper = model_to_train_wrapper[model0_name]
    model1_train_wrapper = model_to_train_wrapper[model1_name]

    model0_kwargs = {
        'my_stream': stream0,
        'sync_info': sync_info,
        'tid': 0,
        'num_epochs': num_epochs,
        'device': device,
        'model_config': config[model0_name]
    }
    model1_kwargs = {
        'my_stream': stream1,
        'sync_info': sync_info,
        'tid': 1,
        'num_epochs': num_epochs,
        'device': device,
        'model_config': config[model1_name]
    }

    start_time = time.time()
    if policy == "tick-tock":
        thread0 = threading.Thread(target=model0_train_wrapper, kwargs=model0_kwargs)
        thread1 = threading.Thread(target=model1_train_wrapper, kwargs=model1_kwargs)
        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()

    elif config['policy'] == "temporal":
        sync_info.no_sync_control = True
        model0_train_wrapper(**model0_kwargs)
        model1_train_wrapper(**model1_kwargs)
    end_time = time.time()
    logging.info(f'It takes {end_time - start_time} seconds in total.')
    notifier.notify(subject='gpu_share training evaluation', body='it ends!')
