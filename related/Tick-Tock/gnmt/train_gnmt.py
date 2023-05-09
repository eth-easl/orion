import os
import time

import torch.nn as nn
import torch.optim
from utils.sync_control import *
import utils

import gnmt.seq2seq.train.trainer as trainers
import gnmt.seq2seq.data.config as config
from gnmt.seq2seq.data.dataset import LazyParallelDataset, RawTextDataset
from gnmt.seq2seq.data.tokenizer import Tokenizer
from gnmt.seq2seq.models.gnmt import GNMT
from gnmt.seq2seq.train.smoothing import LabelSmoothing
from gnmt.seq2seq.utils import setup_seeds
from gnmt.seq2seq.inference.beam_search import SequenceGenerator


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


def eval_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    # build tokenizer
    wmt16_en_de_root = shared_config['wmt16_en_de_root']
    args_vocab = os.path.join(wmt16_en_de_root, 'vocab.bpe.32000')
    args_bpe_codes = os.path.join(wmt16_en_de_root, 'bpe.32000')
    eval_dataset_path = os.path.join(wmt16_en_de_root, 'newstest2014.en')
    args_lang = {'src': 'en', 'tgt': 'de'}

    pad_vocab = pad_vocabulary(model_config['math'])
    tokenizer = Tokenizer(args_vocab, args_bpe_codes, args_lang, pad_vocab)

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
    model = GNMT(**nn_model_config)
    model.type(torch.FloatTensor)
    model = model.to(device)
    model.eval()
    data = RawTextDataset(
        raw_datafile=eval_dataset_path,
        tokenizer=tokenizer,
        sort=False
    )

    batch_size = model_config['batch_size']
    num_requests = shared_config['num_requests']
    num_warm_up_reqs = shared_config['num_warm_up_reqs']
    loader = data.get_loader(
        batch_size=batch_size,
        batch_first=batch_first,
        pad=True,
        repeat=num_requests,
        num_workers=model_config['num_workers']
    )
    beam_size = 5
    max_seq_len = 80
    len_norm_factor = 0.6
    cov_penalty_factor = 0.1
    len_norm_const = 5.0
    insert_target_start = [config.BOS]
    bos = [insert_target_start] * (batch_size * beam_size)
    bos = torch.tensor(bos, dtype=torch.int64, device=device)
    if batch_first:
        bos = bos.view(-1, 1)
    else:
        bos = bos.view(1, -1)

    generator = SequenceGenerator(
        model=model,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        len_norm_factor=len_norm_factor,
        len_norm_const=len_norm_const,
        cov_penalty_factor=cov_penalty_factor
    )

    if beam_size == 1:
        generator_func = generator.greedy_search
    else:
        generator_func = generator.beam_search

    if shared_config['use_dummy_data']:
        logging.info('gnmt uses dummy data')
        try:
            datum = torch.load(model_config['dummy_datum_path'])
            src_single_content, src_single_len, tgt_single_content, tgt_single_len = datum
            src_content = torch.stack([src_single_content for _ in range(batch_size)], dim=1)
            src_len = torch.full((batch_size,), src_single_len)
        except:
            logging.info("Can't load dummy_datum_path; build dummy data on the fly")
            dataloader_iter = iter(loader)
            src, tgt = next(dataloader_iter)
            src_content, src_len = src

        src_content = src_content.to(device)
        src_len = src_len.to(device)
        # indices not used by model in eval()
        # so we can set it as garbage value to align DummyDataLoader with loader
        indices = 0
        virtual_loader = utils.DummyDataLoader(batch=((src_content, src_len), indices))
    else:
        virtual_loader = loader

    virtual_loader_iterator = iter(virtual_loader)


    def eval():
        src, indices = next(virtual_loader_iterator)
        src, src_length = src
        src = src.to(device)
        src_length = src_length.to(device)
        context = model.encode(src, src_length)
        context = [context, src_length.clone(), None]
        _ = generator_func(batch_size, bos, context)

    utils.measure(eval, num_requests, num_warm_up_reqs, tid, shared_config, my_stream, sync_info)


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
    batch_size = model_config['batch_size']
    train_loader = train_data.get_loader(batch_size=batch_size,
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
        keep_checkpoints=5,  # however not used
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
        try:
            datum = torch.load(model_config['dummy_datum_path'])
            src_single_content, src_single_len, tgt_single_content, tgt_single_len = datum
            src_content = torch.stack([src_single_content for _ in range(batch_size)], dim=1)
            tgt_content = torch.stack([tgt_single_content for _ in range(batch_size)], dim=1)
            src_len = torch.full((batch_size,), src_single_len)
            tgt_len = torch.full((batch_size,), tgt_single_len)
        except:
            logging.info("Can't load dummy_datum_path; build dummy data on the fly")
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
            my_stream.synchronize()
            sync_info.pre_measurement_prep(tid)
            # start timer
            start_time = time.time()

        trainer.model.zero_grad()
        trainer.iterate(src, tgt, thread_id=tid, sync_info=sync_info, my_stream=my_stream,
                        batch_idx=batch_idx)

        if batch_idx == num_iterations - 1:
            # reached the last iteration
            break
    my_stream.synchronize()
    duration = time.time() - start_time
    sync_info.post_measurement_prep(tid)
    sync_info.write_kv(f'duration{tid}', duration)
    logging.info(f'tid {tid} it takes {duration} seconds to train gnmt')
    return duration
