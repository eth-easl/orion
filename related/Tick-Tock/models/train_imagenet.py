import torch
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from train_info import TrainInfo


def setup(train_info: TrainInfo, device):
    torch.cuda.set_device(device)
    model = models.__dict__[train_info.arch](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, train_info.optimizer)
    optimizer = optimizer_func(model.parameters(), lr=0.1)

    metric_fn = F.cross_entropy

    if train_info.arch == 'inception_v3':
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
        datasets.ImageFolder(train_info.train_dir, transform=train_transform)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_info.batchsize, sampler=train_sampler, num_workers=train_info.num_workers)

    return model, optimizer, train_loader, metric_fn
