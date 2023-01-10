class TrainInfo:
    def __init__(self, arch, batchsize, num_workers, optimizer, train_dir) -> None:

        self.arch = arch
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.train_dir = train_dir