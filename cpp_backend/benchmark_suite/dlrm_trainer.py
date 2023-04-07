import torch
import threading
import time

from dlrm.model.distributed import DistributedDlrm

def dlrm_loop(batchsize, loader, local_rank, barrier, tid):


    barrier.wait()
    model_config = {}
    
    numerical_features = torch.ones((batchsize, 13)).to(torch.float32).cuda()
    categorical_features = torch.ones((batchsize, 26)).to(torch.long).cuda()
    #click = torch.ones((batchsize)).to(torch.float32).cuda()
    
    torch.cuda.profiler.cudart().cudaProfilerStart()

    model = DistributedDlrm(
        vectors_per_gpu=[27],
        embedding_device_mapping=[[19, 0, 21, 9, 20, 10, 22, 11, 1, 4, 2, 23, 14, 3, 6, 13, 7, 17, 15, 24, 8, 25, 18, 12, 5, 16]],
        embedding_type='joint',
        embedding_dim=64,
        world_num_categorical_features=26,
        categorical_feature_sizes=[8165896, 7912889, 7156453, 5554114, 2675940, 582469, 302516, 245828, 33823, 20046, 17139
            , 12022, 10667, 7339, 7105, 2209, 1382, 968, 104, 97, 63, 35, 15, 11, 4, 4],
        num_numerical_features=13,
        hash_indices=False,
        bottom_mlp_sizes=[256, 128, 64],
        top_mlp_sizes=[1024, 1024, 512, 256, 1],                                        
        interaction_op="dot",
        fp16=False,
        use_cpp_mlp=False,
        bottom_features_ordered=False,
        device='cuda'
    )
    model.eval()

    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        batch_idx = 0
        torch.cuda.synchronize()
                                                                                                                        

        while batch_idx < 1:                                                                                         
            print(f"submit!, batch_idx is {batch_idx}")

            #torch.cuda.profiler.cudart().cudaProfilerStart()
            with torch.no_grad():
                output = model(numerical_features, categorical_features)

            torch.cuda.profiler.cudart().cudaProfilerStop()
            batch_idx += 1                                                                                                                                                                              
        
    print("Epoch took: ", time.time()-start)

#dlrm_loop(16384)
