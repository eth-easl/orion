import os
import time

import torch.nn as nn
import torch.optim
from utils.sync_control import *
import utils

import gnmt.seq2seq.train.trainer as trainers
import gnmt.seq2seq.data.config as config
from gnmt.seq2seq.data.dataset import LazyParallelDataset
from gnmt.seq2seq.data.tokenizer import Tokenizer
from gnmt.seq2seq.models.gnmt import GNMT
from gnmt.seq2seq.train.smoothing import LabelSmoothing
from gnmt.seq2seq.utils import setup_seeds


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


def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    # build tokenizer
    wmt16_en_de_root = shared_config['wmt16_en_de_root']
    args_vocab = os.path.join(wmt16_en_de_root, 'vocab.bpe.32000')
    args_bpe_codes = os.path.join(wmt16_en_de_root, 'bpe.32000')
    args_train_src = os.path.join(wmt16_en_de_root, 'train.tok.clean.bpe.32000.en')
    args_train_tgt = os.path.join(wmt16_en_de_root, 'train.tok.clean.bpe.32000.de')
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
    _, shuffling_seeds = setup_seeds(master_seed=None, epochs=1, device=device)
    train_loader = train_data.get_loader(batch_size=model_config['batch_size'],
                                         seeds=shuffling_seeds,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         batching='bucketing',
                                         batching_opt=batching_opt,
                                         num_workers=model_config['num_workers'])

    total_train_iters = len(train_loader)

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

    num_iterations = model_config['num_iterations']
    warm_up_iters = model_config['warm_up_iters']

    if shared_config['use_dummy_data']:
        logging.info('gnmt uses dummy data')
        train_dataloader_iter = iter(train_loader)
        src, tgt = next(train_dataloader_iter)
        src_content, src_len = src
        tgt_content, tgt_len = tgt
        src_content = src_content.to(device)
        tgt_content = tgt_content.to(device)
        src_len = src_len.to(device)
        tgt_len = tgt_len.to(device)

        virtual_loader = utils.DummyDataLoader(batch=(
            (src_content, src_len),
            (tgt_content, tgt_len)
        ))
    else:
        virtual_loader = train_loader
    logging.info(f'gnmt is set up with {num_iterations}')
    for batch_idx, (src, tgt) in enumerate(virtual_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            torch.cuda.synchronize(device)
            sync_info.pre_measurement_prep(tid)
            # start timer
            start_time = time.time()

        trainer.model.zero_grad()
        trainer.iterate(src, tgt, thread_id=tid, sync_info=sync_info, my_stream=my_stream,
                        batch_idx=batch_idx)

        if batch_idx == num_iterations - 1:
            # reached the last iteration
            break
    torch.cuda.synchronize(device)
    sync_info.post_measurement_prep(tid)
    duration = time.time() - start_time
    logging.info(f'tid {tid} it takes {duration} seconds to train gnmt')
    return duration
