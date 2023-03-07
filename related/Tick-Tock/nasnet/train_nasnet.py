import time
from torchvision import models, datasets, transforms
import torch
import torch.nn.functional as F
from nasnet.nasnet import NASNetALarge
from nasnet.nasnet_mobile import NASNetAMobile
from utils.sync_info import SyncInfo
from utils.sync_control import *
from utils.constants import *


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
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
        datasets.ImageFolder(imagenet_root, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=model_config['num_workers'])
    metric_fn = F.cross_entropy
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.1)

    print(f"training {tid} starts!!")
    loss_sum = 0
    print_every = 50


    with TrainingControl(sync_info=sync_info, device=device), torch.cuda.stream(my_stream):
        start = time.time()
        for _ in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                    data, target = batch[0].to(device), batch[1].to(device)
                    output = model(data)
                    loss = metric_fn(output, target)
                    loss_sum += loss.item()

                if batch_idx % print_every == 0:
                    print(f"loss for thread {tid}: {loss_sum / print_every}")
                    loss_sum = 0

                with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

    end = time.time()
    print(f"TID: {tid}, training took {end - start} sec.")
