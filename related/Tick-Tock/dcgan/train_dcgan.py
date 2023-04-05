import time
from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
from dcgan.dcgan import *
import random
from utils.sync_control import *

# code from https://github.com/pytorch/examples/blob/main/dcgan/main.py

def setup_dataloader(model_config, shared_config):
    dataset_name = model_config['dataset']
    input_image_size = model_config['input_image_size']
    if dataset_name == 'imagenet':
        dataset = datasets.ImageFolder(
            root=shared_config['imagenet_root'],
            transform=transforms.Compose([
                transforms.Resize(input_image_size),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        num_channels = 3
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=shared_config['cifar10_root'],
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        num_channels = 3
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=shared_config['mnist_root'],
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
        num_channels = 1
    else:
        raise ValueError(f'unsupported dataset {dataset_name}')
    batch_size = model_config['batch_size']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=model_config['num_workers'])
    return dataloader, num_channels


def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    seed = int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)
    dataloader, num_channels = setup_dataloader(model_config, shared_config)
    latent_z_vec_size = model_config['latent_z_vec_size']
    netG = Generator(
        ngf=model_config['num_gen_filters'],
        nc=num_channels,
        nz=latent_z_vec_size
    ).to(device)
    netG.apply(weights_init)

    netD = Discriminator(
        ndf=model_config['num_dis_filters'],
        nc=num_channels
    ).to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizerD = optimizer_func(netD.parameters())
    optimizerG = optimizer_func(netG.parameters())
    batch_size = model_config['batch_size']


    for batch_idx, batch in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                real_images = batch[0].to(device)
                label = torch.full((batch_size,), real_label, dtype=real_images.dtype, device=device)
                output = netD(real_images)

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                errD_real = criterion(output, label)
                errD_real.backward()


        # train discriminator with fake data
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                noise = torch.randn(batch_size, latent_z_vec_size, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                errD_fake = criterion(output, label)
                errD_fake.backward()
                optimizerD.step()
                netD.zero_grad()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                label.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()
                netG.zero_grad()
