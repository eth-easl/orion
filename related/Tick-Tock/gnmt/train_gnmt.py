import os
import time

import torch.nn as nn
import torch.optim
from utils.sync_info import SyncInfo
from utils.sync_control import *
import utils.constants as constants


import gnmt.seq2seq.train.trainer as trainers
import gnmt.seq2seq.data.config as config
from gnmt.seq2seq.data.dataset import LazyParallelDataset
from gnmt.seq2seq.data.tokenizer import Tokenizer
from gnmt.seq2seq.models.gnmt import GNMT
from gnmt.seq2seq.train.smoothing import LabelSmoothing


def pad_vocabulary(math):
    if math == 'tf32' or math == 'fp16' or math == 'manual_fp16':
        pad_vocab = 8
    elif math == 'fp32':
        pad_vocab = 1
    return pad_vocab


def build_criterion(padding_idx, smoothing):
    if smoothing == 0.:
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)
    return criterion


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    # build tokenizer
    args_vocab = os.path.join(constants.wmt16_en_de_root, 'vocab.bpe.32000')
    args_bpe_codes = os.path.join(constants.wmt16_en_de_root, 'bpe.32000')
    args_train_src = os.path.join(constants.wmt16_en_de_root, 'train.tok.clean.bpe.32000.en')
    args_train_tgt = os.path.join(constants.wmt16_en_de_root, 'train.tok.clean.bpe.32000.de')
    args_lang = {'src': 'en', 'tgt': 'de'}

    pad_vocab = pad_vocabulary(model_config['math'])
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
    criterion = build_criterion(config.PAD, 0.1).to(device)
    opt_config = {'optimizer': 'Adam', 'lr': 2.00e-3}
    # get data loaders
    batching_opt = {'shard_size': 80, 'num_buckets': 5}
    train_loader = train_data.get_loader(batch_size=model_config['batch_size'],
                                         seeds=int(time.time()),
                                         batch_first=batch_first,
                                         shuffle=True,
                                         batching='bucketing',
                                         batching_opt=batching_opt,
                                         num_workers=model_config['num_workers'])

    total_train_iters = len(train_loader) // num_epochs

    trainer_options = dict(
        model=model,
        criterion=criterion,
        grad_clip=5.0,
        iter_size=1,
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

    with TrainingControl(sync_info=sync_info, device=device), torch.cuda.stream(my_stream):
        for epoch in range(num_epochs):
            train_loader.sampler.set_epoch(epoch)
            for batch_idx, (src, tgt) in enumerate(train_loader):
                trainer.model.zero_grad()
                trainer.iterate(src, tgt, thread_id=tid, sync_info=sync_info, my_stream=my_stream, batch_idx=batch_idx)


