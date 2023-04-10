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
        "openimages-mlperf": (get_openimages, 264, None),  # L0 classes with more than 1000 samples
    }
    return paths[name]


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def collate_fn(batch):
    return tuple(zip(*batch))

dataset_to_num_classes = {
    'coco': 91,
    'openimages': 601,
    'openimages-mlperf': 264
}


def setup(model_config, shared_config, device):
    data_layout = "channels_last"
    batch_size = model_config['batch_size']
    model = retinanet_from_backbone(backbone='resnext50_32x4d',
                                    num_classes=dataset_to_num_classes[model_config['dataset_name']],
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

    model.train()

    if shared_config['use_dummy_data']:
        images = [torch.ones((3, 768, 1024)).to(torch.float32) for _ in range(batch_size)]
        targets = [
            {
                'boxes': torch.tensor([[3.8400, 42.2873, 597.1200, 660.5751],
                                       [367.3600, 2.5626, 1008.6400, 682.3594]]),
                'labels': torch.tensor([148, 257]),
                'image_id': torch.tensor([299630]),
                'area': torch.tensor([366817.7812, 435940.0625]),
                '   iscrowd': torch.tensor([0, 0]),
            }
            for _ in range(batch_size)
        ]
        virtual_loader = utils.DummyDataLoader(batch=(images, targets))
    else:
        dataset_fn, num_classes, data_path = get_dataset_fn(model_config['dataset_name'], shared_config)
        dataset = dataset_fn(name=model_config['dataset_name'],
                             root=data_path,
                             image_set="train",
                             transforms=get_transform(True, 'hflip'))
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=model_config['num_workers'],
            pin_memory=False, collate_fn=collate_fn)
        virtual_loader = data_loader

    return model, virtual_loader, scaler, optimizer

def eval_wrapper(sync_info: BasicSyncInfo, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    model, data_loader, _, _ = setup(model_config, shared_config, device)
    model.eval()
    num_requests = shared_config['num_requests']
    num_warm_up_reqs = shared_config['num_warm_up_reqs']

    loader_iterator = iter(data_loader)

    def eval():
        images, targets = next(loader_iterator)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model(images, targets)

    utils.measure(eval, num_requests, num_warm_up_reqs, tid, shared_config, my_stream, sync_info)


def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)

    model, virtual_loader, scaler, optimizer = setup(model_config, shared_config, device)

    num_iterations = model_config['num_iterations']
    warm_up_iters = model_config['warm_up_iters']

    logging.info(f'retinanet is set up with {num_iterations}')

    for batch_idx, (images, targets) in enumerate(virtual_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            my_stream.synchronize()
            sync_info.pre_measurement_prep(tid)
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

    my_stream.synchronize()
    duration = time.time() - start_time
    sync_info.post_measurement_prep(tid)
    sync_info.write_kv(f'duration{tid}', duration)
    logging.info(f'tid {tid} it takes {duration} seconds to train retinanet')
    return duration
