import torch
import threading
import time

from model.retinanet import retinanet_from_backbone 

def retinanet_loop(batchsize, loader, local_rank, barrier, tid):

    barrier.wait()

    images = [torch.ones((3,768,1024)).to(torch.float32).cuda() for _ in range(batchsize)] 
    
    torch.cuda.profiler.cudart().cudaProfilerStart()

    model = retinanet_from_backbone(
            backbone="resnext50_32x4d",
            num_classes=264,
            image_size=[800, 800],
            data_layout='channels_last',
            pretrained=False,
            trainable_backbone_layers=3).cuda()

    model.eval()

    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        batch_idx = 0
        torch.cuda.synchronize()
                                                                                                
        while batch_idx < 1:
            print(f"submit!, batch_idx is {batch_idx}")
            with torch.no_grad():
                output = model(images)                                                                                  
            torch.cuda.profiler.cudart().cudaProfilerStop()
            #print(output)                                                                                                                                                                                                 
            batch_idx += 1  
        
        while True:
            pass
