import torch
import threading

import yaml
from utils.sync_info import SyncInfo
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

if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

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
    sync_info = SyncInfo(eventf0, eventb0, eventf1, eventb1, barrier=threading.Barrier(2))
    model0_name = config['model0']
    model1_name = config['model1']
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

    if config['policy'] == "tick-tock":
        thread0 = threading.Thread(target=model0_train_wrapper, kwargs=model0_kwargs)
        thread1 = threading.Thread(target=model1_train_wrapper, kwargs=model1_kwargs)
        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()
        print("All threads joined!!!!!!!!!!")

    elif config['policy'] == "temporal":
        sync_info.no_sync_control = True
        model0_train_wrapper(**model0_kwargs)
        model1_train_wrapper(**model1_kwargs)
