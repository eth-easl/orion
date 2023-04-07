import torch
import threading
import time

from model.retinanet import retinanet_from_backbone

def retinanet_loop(batchsize, train, local_rank, barriers, tid):

    barriers[0].wait()

    model = retinanet_from_backbone(
            backbone="resnext50_32x4d",
            num_classes=264,
            image_size=[800, 800],
            data_layout='channels_last',
            pretrained=False,
            trainable_backbone_layers=3).cuda()
    images = [torch.ones((3,768,1024)).to(torch.float32).cuda() for _ in range(batchsize)]
    targets = [
        {
            'boxes': torch.tensor([[   3.8400,   42.2873,  597.1200,  660.5751],
                            [ 367.3600, 2.5626, 1008.6400,  682.3594]]).cuda(),
            'labels': torch.tensor([148, 257]).cuda(),
            'image_id':  torch.tensor([299630]).cuda(),
            'area': torch.tensor([366817.7812, 435940.0625]).cuda(),
            'iscrowd': torch.tensor([0, 0]).cuda(),
        }
        for _ in range(batchsize)
    ]

    if train:
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.1)
    else:
        model.eval()

    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        batch_idx = 0

        while batch_idx < 10:
            print(f"submit!, batch_idx is {batch_idx}")
            if train:
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    output = model(images)
            #print(output)
            batch_idx += 1

            print("sent everything!")

            if (batch_idx == 1) and train: # for backward
                barriers[0].wait()

            #barriers[0].wait()
            if batch_idx < 10:
                barriers[0].wait()
                #barriers[0].wait()

        print("Finished! Ready to join!")