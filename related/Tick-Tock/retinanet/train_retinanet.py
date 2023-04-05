import torch
import torch.utils.data
import numpy as np
import retinanet.presets as presets
import time
from utils.sync_control import *
from retinanet.model.retinanet import retinanet_from_backbone
from retinanet.coco_utils import get_openimages, get_coco
import utils

def get_dataset_fn(name, shared_config):
    paths = {
        "coco": (get_coco, 91, shared_config['coco_root']),
        "openimages": (get_openimages, 601, None),  # Full openimages dataset
        "openimages-mlperf": (get_openimages, None),  # L0 classes with more than 1000 samples
    }
    return paths[name]


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def collate_fn(batch):
    return tuple(zip(*batch))


def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    dataset_fn, num_classes, data_path = get_dataset_fn(model_config['dataset_name'], shared_config)
    data_layout = "channels_last"
    batch_size = model_config['batch_size']
    model = retinanet_from_backbone(backbone='resnext50_32x4d',
                                    num_classes=num_classes,
                                    image_size=[800, 800],
                                    data_layout=data_layout,
                                    pretrained=False,
                                    trainable_backbone_layers=3)
    model.to(device)
    if data_layout == 'channels_last':
        model = model.to(memory_format=torch.channels_last)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=model_config['use_amp'])

    dataset = dataset_fn(name=model_config['dataset_name'],
                         root=data_path,
                         image_set="train",
                         transforms=get_transform(True, 'hflip'))
    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=model_config['num_workers'],
        pin_memory=False, collate_fn=collate_fn)

    model.train()

    num_iterations = model_config['num_iterations']
    warm_up_iters = model_config['warm_up_iters']
    if shared_config['use_dummy_data']:
        train_dataloader_iter = iter(data_loader)
        images, targets = next(train_dataloader_iter)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        virtual_loader = utils.DummyDataLoader(batch=(images, targets))
    else:
        virtual_loader = data_loader

    logging.info(f'retinat is set up with {num_iterations}')

    for batch_idx, (images, targets) in enumerate(virtual_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            torch.cuda.synchronize(device)
            if not sync_info.no_sync_control:
                sync_info.barrier.wait()
            # start timer
            start_time = time.time()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if batch_idx == num_iterations - 1:
            # reached the last iteration
            break

    sync_info.no_sync_control = True
    torch.cuda.synchronize(device)

    duration = time.time() - start_time
    logging.info(f'tid {tid} it takes {duration} seconds to train retinanet')
    return duration
