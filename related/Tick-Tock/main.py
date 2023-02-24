import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import threading

import yaml
from utils.sync_info import SyncInfo
from vision.train_imagenet import train_wrapper as vision_train_wrapper
from nasnet.train_nasnet import train_wrapper as nasnet_train_wrapper

model2train_wrapper = {
    'nasnet': nasnet_train_wrapper,
    'vision': vision_train_wrapper,
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

    sync_info = SyncInfo(eventf0, eventb0, eventf1, eventb1)
    model0train_wrapper = model2train_wrapper[config['model0']['name']]
    model1train_wrapper = model2train_wrapper[config['model1']['name']]
    if config['policy'] == "tick-tock":
        thread0 = threading.Thread(target=model0train_wrapper, kwargs={
            'my_stream': stream0,
            'sync_info': sync_info,
            'tid': 0,
            'num_epochs': 5,
            'device': device,
            'model_config': config['model0']
        })
        thread1 = threading.Thread(target=model1train_wrapper, kwargs={
            'my_stream': stream1,
            'sync_info': sync_info,
            'tid': 1,
            'num_epochs': 5,
            'device': device,
            'model_config': config['model1']
        })
        thread0.start()
        thread1.start()

        thread0.join()
        thread1.join()
        print("All threads joined!!!!!!!!!!")

    elif config['policy'] == "temporal":
        sync_info.no_sync_control = True
        model0train_wrapper(my_stream=stream0, sync_info=sync_info, tid=0, num_epochs=5, device=device,
                            model_config=config['model0'])
        model1train_wrapper(my_stream=stream1, sync_info=sync_info, tid=1, num_epochs=5, device=device,
                            model_config=config['model1'])
