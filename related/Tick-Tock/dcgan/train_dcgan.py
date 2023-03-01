import time
from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
from dcgan.dcgan import *

import utils.constants as constants
from utils.sync_info import SyncInfo
from utils.sync_control import *

# code from https://github.com/pytorch/examples/blob/main/dcgan/main.py

def setup_dataloader(model_config):
    dataset_name = model_config['dataset']
    input_image_size = model_config['input_image_size']
    if dataset_name == 'imagenet':
        dataset = datasets.ImageFolder(
            root=constants.imagenet_root,
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
            root=constants.cifar10_root,
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
            root=constants.mnist_root,
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
                                             shuffle=True, num_workers=2)
    return dataloader, num_channels


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    dataloader, num_channels = setup_dataloader(model_config)
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
    print_every = 50

    with TrainingControl(sync_info=sync_info, device=device), torch.cuda.stream(my_stream):
        start = time.time()
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                with ForwardControl(thread_id=tid, sync_info=sync_info, stream=my_stream):
                    netD.zero_grad()
                    real_images = batch[0].to(device)
                    label = torch.full((batch_size,), real_label, dtype=real_images.dtype, device=device)
                    output = netD(real_images)

                with BackwardControl(thread_id=tid, sync_info=sync_info, stream=my_stream):
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()

                # train discriminator with fake data
                with ForwardControl(thread_id=tid, sync_info=sync_info, stream=my_stream):
                    noise = torch.randn(batch_size, latent_z_vec_size, 1, 1, device=device)
                    fake = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake.detach())

                with BackwardControl(thread_id=tid, sync_info=sync_info, stream=my_stream):
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    errD = errD_real + errD_fake
                    optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                with ForwardControl(thread_id=tid, sync_info=sync_info, stream=my_stream):
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    output = netD(fake)

                with BackwardControl(thread_id=tid, sync_info=sync_info, stream=my_stream):
                    errG = criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    optimizerG.step()

                if batch_idx % print_every == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, batch_idx, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    end = time.time()
    print(f"TID: {tid}, training took {end - start} sec.")
