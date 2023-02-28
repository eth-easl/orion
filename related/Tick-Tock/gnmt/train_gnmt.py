import os
import torch.nn as nn
import torch.optim
from utils.sync_info import SyncInfo
import gnmt.seq2seq.utils as seq2seq_utils

import gnmt.seq2seq.train.trainer as trainers
import gnmt.seq2seq.utils as utils
import gnmt.seq2seq.data.config as config
from gnmt.seq2seq.data.dataset import LazyParallelDataset
from gnmt.seq2seq.data.tokenizer import Tokenizer
from gnmt.seq2seq.models.gnmt import GNMT
from gnmt.seq2seq.train.smoothing import LabelSmoothing

DATASET_DIR = '/cluster/scratch/xianma/wmt16'
args_vocab = os.path.join(DATASET_DIR, 'vocab.bpe.32000')
args_bpe_codes = os.path.join(DATASET_DIR, 'bpe.32000')
args_train_src = os.path.join(DATASET_DIR, 'train.tok.clean.bpe.32000.en')
args_train_tgt = os.path.join(DATASET_DIR, 'train.tok.clean.bpe.32000.de')
args_lang = {'src': 'en', 'tgt': 'de'}


def set_iter_size(train_iter_size, train_global_batch_size, train_batch_size):
    """
    Automatically set train_iter_size based on train_global_batch_size,
    world_size and per-worker train_batch_size

    :param train_global_batch_size: global training batch size
    :param train_batch_size: local training batch size
    """
    if train_global_batch_size is not None:
        global_bs = train_global_batch_size
        bs = train_batch_size
        world_size = utils.get_world_size()
        assert global_bs % (bs * world_size) == 0
        train_iter_size = global_bs // (bs * world_size)
    return train_iter_size


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)
    return criterion


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    # build tokenizer
    pad_vocab = seq2seq_utils.pad_vocabulary(model_config['math'])
    tokenizer = Tokenizer(args_vocab, args_bpe_codes, args_lang, pad_vocab)
    # build datasets
    train_data = LazyParallelDataset(
        src_fname=args_train_src,
        tgt_fname=args_train_tgt,
        tokenizer=tokenizer,
        min_len=0,
        max_len=50,
        sort=False,
        max_size=None,
    )

    # build GNMT model
    vocab_size = tokenizer.vocab_size
    batch_first = False
    nn_model_config = {'hidden_size': 1024,
                    'vocab_size': vocab_size,
                    'num_layers': 4,
                    'dropout': 0.2,
                    'batch_first': batch_first,
                    'share_embedding': True,
                    }
    model = GNMT(**nn_model_config).to(device)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, 0.1).to(device)
    opt_config = {'optimizer': 'Adam', 'lr': 2.00e-3}
    worker_seeds, shuffling_seeds = utils.setup_seeds(None, num_epochs, device)
    # get data loaders
    batching_opt = {'shard_size': 80, 'num_buckets': 5}
    train_loader = train_data.get_loader(batch_size=model_config['batch_size'],
                                         seeds=shuffling_seeds,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         batching='bucketing',
                                         batching_opt=batching_opt,
                                         num_workers=2)
    train_iter_size = set_iter_size(train_iter_size=1, train_global_batch_size=None, train_batch_size=128)
    total_train_iters = len(train_loader) // train_iter_size * num_epochs

    trainer_options = dict(
        model=model,
        criterion=criterion,
        grad_clip=5.0,
        iter_size=train_iter_size,
        save_dir=None,
        save_freq=None,
        save_info=None,
        opt_config=opt_config,
        scheduler_config={'warmup_steps': 200,
                          'remain_steps': 0.666,
                          'decay_interval': None,
                          'decay_steps': 4,
                          'decay_factor': 0.5},
        train_iterations=total_train_iters,
        keep_checkpoints=5, # however not used
        math=model_config['math'],
        loss_scaling={
            'init_scale': 8192,
            'upscale_interval': 128
        },
        print_freq=10,
        intra_epoch_eval=0,
        translator=None,
        prealloc_mode='off',
        warmup=None,
    )

    trainer = trainers.Seq2SeqTrainer(**trainer_options)
    torch.set_grad_enabled(True)
    trainer.model.train()
    print_every = 50
    loss_sum = 0
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch_idx, (src, tgt) in enumerate(train_loader):
            trainer.model.zero_grad()
            loss_per_token, loss_per_sentence = trainer.iterate(src, tgt, thread_id=tid, sync_info=sync_info, my_stream=my_stream)
            loss_sum += loss_per_token
            if batch_idx % print_every == 0:
                print(f'loss: {loss_sum / print_every}')
                loss_sum = 0

    sync_info.no_sync_control = True

