import torch
import threading
import time
import numpy as np

from model.retinanet import retinanet_from_backbone

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.images = [torch.ones((3,768,1024)).to(torch.float32) for _ in range(self.batchsize)]
        self.targets = [
            {
                'boxes': torch.tensor([[   3.8400,   42.2873,  597.1200,  660.5751],
                            [ 367.3600, 2.5626, 1008.6400,  682.3594]]),
                'labels': torch.tensor([148, 257]),
                'image_id':  torch.tensor([299630]),
                'area': torch.tensor([366817.7812, 435940.0625]),
                'iscrowd': torch.tensor([0, 0]),
            }
            for _ in range(self.batchsize)
        ]

    def __iter__(self):
        return self

    def __next__(self):
        return self.images, self.targets


def retinanet_loop(batchsize, train, default, num_iters, rps, dummy_data, local_rank, start_barriers, end_barriers, tid):

    start_barriers[tid].wait()

    timings = []

    if default:
        s = torch.cuda.default_stream()
    else:
        s = torch.cuda.Stream()

    if rps > 0:
        sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters


    model = retinanet_from_backbone(
            backbone="resnext50_32x4d",
            num_classes=264,
            image_size=[800, 800],
            data_layout='channels_last',
            pretrained=False,
            trainable_backbone_layers=3).cuda()

    if train:
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.1)
    else:
        model.eval()

    train_loader = DummyDataLoader(batchsize)
    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)

    start_barriers[tid].wait()

    with torch.cuda.stream(s):
        for i in range(1):
            print("Start epoch: ", i)

            while batch_idx < num_iters:
                print(f"submit!, batch_idx is {batch_idx}")
                start = time.time()

                if train:
                    images, targets = batch
                    gpu_images = [x.to(local_rank) for x in images]
                    gpu_targets = [({k:v.to(local_rank) for k,v in x.items()}) for x in targets]
                    optimizer.zero_grad()
                    loss_dict = model(gpu_images, gpu_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        images = batch[0]
                        gpu_images = [x.to(local_rank) for x in images]
                        output = model(gpu_images)

                s.synchronize()
                iter_time = time.time()-start
                timings.append(iter_time)

                time.sleep(sleep_times[batch_idx])
                print(f"{batch_idx} finished, took {iter_time} sec, now sleep for {sleep_times[batch_idx]} sec")

                batch_idx, batch = next(train_iter)

                #if batch_idx < num_iters:
                    # barriers[0].wait()
            end_barriers[tid].wait()

    timings = timings[2:]
    p50 = np.percentile(timings, 50)
    p95 = np.percentile(timings, 95)
    p99 = np.percentile(timings, 99)

    print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
