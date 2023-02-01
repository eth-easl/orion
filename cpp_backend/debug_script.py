import torch

def test_func(tid, barrier):
    barrier.wait()

    a = torch.ones(1, device='cuda')
    print(a)
    #b = torch.ones(1, device='cuda')
    #c = torch.ones(1, device='cuda')
    #b.abs()
    #c.abs()
    torch.cuda.synchronize()

    #while True:
    #    pass
