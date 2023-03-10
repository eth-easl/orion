import torch
import torch.utils.data
import numpy as np
import retinanet.presets as presets
from utils.sync_info import SyncInfo
import time
from utils.sync_control import *
from retinanet.model.retinanet import retinanet_from_backbone
from retinanet.coco_utils import get_openimages, get_coco

DATASET_DIR = '/cluster/scratch/xianma/retinanet'


def get_dataset_fn(name):
    paths = {
        "coco": (get_coco, 91, '/cluster/scratch/xianma/retinanet'),
        "openimages": (get_openimages, 601, None),  # Full openimages dataset
        "openimages-mlperf": (get_openimages, 264, None),  # L0 classes with more than 1000 samples
    }
    return paths[name]


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def collate_fn(batch):
    return tuple(zip(*batch))


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    print("Getting dataset information")

    dataset_fn, num_classes, data_path = get_dataset_fn(name=model_config['dataset_name'])
    data_layout = "channels_last"
    batch_size = model_config['batch_size']
    model = retinanet_from_backbone(backbone='resnext50_32x4d',
                                    num_classes=num_classes,
                                    image_size=[800, 800],
                                    data_layout=data_layout,
                                    pretrained=False,
                                    pretrained_backbone=False,
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
        pin_memory=True, collate_fn=collate_fn)

    model.train()
    loss_sum = 0
    print_every = 50
    with TrainingControl(sync_info=sync_info, device=device), torch.cuda.stream(my_stream):
        for epoch in range(num_epochs):
            for batch_idx, (images, targets) in enumerate(data_loader):
                with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.cuda.amp.autocast(enabled=model_config['use_amp']):
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())

                loss_sum += losses.item()
                if batch_idx % print_every == 0:
                    print(f"loss for thread {tid}: {loss_sum / print_every}")
                    loss_sum = 0

                with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
