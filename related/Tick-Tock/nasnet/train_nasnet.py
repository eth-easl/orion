import time
from torchvision import models, datasets, transforms
import torch
import torch.nn.functional as F
from nasnet.nasnet import NASNetALarge
from nasnet.nasnet_mobile import NASNetAMobile
from utils.sync_control import *


def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    arc = model_config['arc']
    model = NASNetALarge(num_classes=1000) if arc == 'large' else NASNetAMobile(num_classes=1000)
    model = model.to(device)
    model.train()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(331 if arc == 'large' else 224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = \
        datasets.ImageFolder(shared_config['imagenet_root'], transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=model_config['num_workers'])
    metric_fn = F.cross_entropy
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.001)

    for batch_idx, batch in enumerate(train_loader):
        data, target = batch[0].to(device), batch[1].to(device)
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                output = model(data)
                loss = metric_fn(output, target)

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
