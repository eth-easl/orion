import torch
import threading
import time
import sys

sys.path.append(f"{os.path.expanduser( '~' )}/mlcommons/single_stage_detector/ssd")
from model.retinanet import retinanet_from_backbone

def retinanet(batchsize, local_rank, do_eval=True, profile=None):

    model = retinanet_from_backbone(
            backbone="resnext50_32x4d",
            num_classes=264,
            image_size=[800, 800],
            data_layout='channels_last',
            pretrained=False,
            trainable_backbone_layers=3).cuda()
    images = [torch.ones((3,768,1024)).to(torch.float32).cuda() for _ in range(batchsize)]
    # just a dummy example
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

    if do_eval:
        model.eval()
    else:
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.1)

    batch_idx = 0

    while batch_idx < 10:

        print(f"run {batch_idx}")

        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()

        if do_eval:
            with torch.no_grad():
                output = model(images)
        else:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()

        batch_idx += 1

    print("Done!")

if __name__ == "__main__":
    retinanet(4, 0, True, 'nsys')
