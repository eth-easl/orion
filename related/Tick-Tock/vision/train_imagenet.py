import torch
from torchvision import models, datasets, transforms
import torch.nn.functional as F

from utils.sync_info import SyncInfo
from utils.sync_control import *
import utils.constants as constants




def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    model, optimizer, train_loader, metric_fn = setup(model_config, device)
    model.train()

    with TrainingControl(sync_info=sync_info, device=device), torch.cuda.stream(my_stream):
        for _ in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                    data, target = batch[0].to(device), batch[1].to(device)
                    output = model(data)
                    loss = metric_fn(output, target)

                with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


def setup(model_config, device):
    torch.cuda.set_device(device)
    arc = model_config['arc']
    model = models.__dict__[arc](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.1)

    metric_fn = F.cross_entropy

    if arc == 'inception_v3':
        train_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = \
        datasets.ImageFolder(constants.imagenet_root, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=model_config['num_workers'])

    return model, optimizer, train_loader, metric_fn
