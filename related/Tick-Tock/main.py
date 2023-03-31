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
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model0_name = config['model0']
    model1_name = config['model1']
    model0_config = config[model0_name]
    model1_config = config[model1_name]
    readable_model0_name = readable_model_name(model0_name, model0_config)
    readable_model1_name = readable_model_name(model1_name, model1_config)
    policy = config['policy']

    if args.log is None:
        args.log = f'{readable_model0_name}-{readable_model1_name}-{policy}.log'

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


    logging.info(f'start training with {model0_name} and {model1_name} using {policy}')
    for key, value in config['shared_config'].items():
        constants.__dict__[key] = value

    device = torch.device("cuda:0")

    stream0 = torch.cuda.Stream(device=device)
    stream1 = torch.cuda.Stream(device=device)

    eventf0 = threading.Event()
    eventb0 = threading.Event()

    eventf1 = threading.Event()
    eventb1 = threading.Event()

    event_cudaf0 = torch.cuda.Event()
    event_cudab0 = torch.cuda.Event()

    event_cudaf1 = torch.cuda.Event()
    event_cudab1 = torch.cuda.Event()

    eventf1.set()  # t0 starts
    eventb1.set()

    num_epochs = config['num_epochs']
    logging.info(f'number of epochs: {num_epochs}')
    sync_info = SyncInfo(eventf0, eventb0, eventf1, eventb1,
                         event_cudaf0, event_cudab0, event_cudaf1, event_cudab1,
                         barrier=threading.Barrier(2))

    model0_train_wrapper = model_to_train_wrapper[model0_name]
    model1_train_wrapper = model_to_train_wrapper[model1_name]

    model0_kwargs = {
        'my_stream': stream0,
        'sync_info': sync_info,
        'tid': 0,
        'num_epochs': num_epochs,
        'device': device,
        'model_config': model0_config
    }
    model1_kwargs = {
        'my_stream': stream1,
        'sync_info': sync_info,
        'tid': 1,
        'num_epochs': num_epochs,
        'device': device,
        'model_config': model1_config
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
    # if constants.enable_profiling:
    #     torch.cuda.cudart().cudaProfilerStop()
    end_time = time.time()
    logging.info(f'It takes {end_time - start_time} seconds in total.')
    notifier.notify(
        subject=f'The experiment training {readable_model0_name} and {readable_model1_name} with {policy} is finished!',
        body=utils.dict2pretty_str(config)
    )
