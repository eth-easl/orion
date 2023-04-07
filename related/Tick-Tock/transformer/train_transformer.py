import functools
import warnings

import yaml
import utils
from utils.sync_control import *
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import yaml
import transformer.lamb as lamb
from transformer.data_utils import *
from transformer.mem_transformer import MemTransformerLM
import sys
try:
    from apex import amp
except ModuleNotFoundError:
    warnings.warn('APEX AMP is unavailable')


dataset = 'wt103'
vocab = 'word'


def init_weight(weight, model_consts):
    # if init == 'uniform':
    #     nn.init.uniform_(weight, -0.1, 0.1)
    # elif init == 'normal':
    nn.init.normal_(weight, 0.0, model_consts['init_std'])


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, model_consts):
    proj_init_std = 0.01
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, model_consts)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, model_consts)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, model_consts)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, proj_init_std)
        if hasattr(m, 'out_layers_weights'):
            for i in range(len(m.out_layers_weights)):
                if m.out_layers_weights[i] is not None:
                    init_weight(m.out_layers_weights[i], model_consts)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, model_consts['init_std'])
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb, model_consts)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, model_consts)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, model_consts)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def setup(model_config, shared_config, device):
    # Before we do anything with models, we want to ensure that we get fp16
    # execution of torch.einsum in APEX AMP.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--apex_amp_opt_level O2` will remove the need for this
    # code, but it is still valid.
    if 'apex' in sys.modules:
        amp.register_half_function(torch, 'einsum')

    seed = int(time.time())
    arch = model_config['arch']
    with open('./transformer/transformer_consts.yaml', 'r') as file:
        model_consts = yaml.load(file, Loader=yaml.FullLoader)[arch]

    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = model_config['batch_size']
    corpus = get_lm_corpus(datadir=shared_config['wikitext_103_dir'], dataset='wt103', vocab=model_consts['vocab'])
    ntokens = len(corpus.vocab)

    ext_len = 0
    tr_iter = corpus.get_iterator('train', batch_size, model_consts['tgt_len'], device=device, ext_len=ext_len)

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if model_consts['adaptive']:
        cutoffs = [19997, 39997, 199997]
        tie_projs += [True] * len(cutoffs)
    sample_softmax = -1
    MemTransformerLM_kwargs = {
        'n_token': ntokens,
        'n_layer': model_consts['n_layer'],
        'n_head': model_consts['n_head'],
        'd_model': model_consts['d_model'],
        'd_head': model_consts['d_head'],
        'd_inner': model_consts['d_inner'],
        'dropout': model_consts['dropout'],
        'dropatt': model_consts['dropatt'],
        'dtype': None,
        'tie_weight': True,
        'd_embed': model_consts['d_model'],
        'div_val': model_consts['div_val'],
        'tie_projs': tie_projs,
        'pre_lnorm': False,
        'tgt_len': model_consts['tgt_len'],
        'ext_len': ext_len,
        'mem_len': model_consts['mem_len'],
        'cutoffs': cutoffs,
        'same_length': False,
        'attn_type': 0,
        'clamp_len': -1,
        'sample_softmax': sample_softmax,
    }
    model = MemTransformerLM(**MemTransformerLM_kwargs)
    model.apply(functools.partial(weights_init, model_consts=model_consts))
    # ensure embedding init is not overridden by out_layer in case of weight sharing
    model.word_emb.apply(functools.partial(weights_init, model_consts=model_consts))

    # jitlamb optimizer
    optimizer = lamb.JITLamb(model.parameters(), lr=model_consts['lr'], weight_decay=0.0)

    model = model.to(device)
    scaler = None
    if model_config['use_fp16']:
        if model_config['amp'] == 'pytorch':
            scaler = torch.cuda.amp.GradScaler()
        elif model_config['amp'] == 'apex':
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=model_config['apex_amp_opt_level'],
            )

    train_iter = tr_iter.get_fixlen_iter()
    if shared_config['use_dummy_data']:
        first_batch = next(train_iter)
        for t in first_batch:
            if torch.is_tensor(t):
                logging.info(f'part of dummy data shape: {t.shape}')
        virtual_loader = utils.DummyDataLoader(batch=first_batch)
    else:
        virtual_loader = train_iter

    return model, virtual_loader, optimizer


def eval_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    model, data_loader, _ = setup(model_config, shared_config, device)
    model.eval()

    num_requests = shared_config['num_requests']
    num_warm_up_reqs = shared_config['num_warm_up_reqs']

    loader_iterator = iter(data_loader)

    mems = None
    def eval():
        nonlocal mems
        data, target, _, _ = next(loader_iterator)
        _, mems = model(data, target, mems)

    utils.measure(eval, num_requests, num_warm_up_reqs, tid, shared_config, my_stream, sync_info)


def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    model, data_loader, optimizer = setup(model_config, shared_config, device)

    model.train()

    enable_autocast = model_config['use_fp16'] and model_config['amp'] == 'pytorch'
    mem = None
    clip = 0.25

    num_iterations = model_config['num_iterations']
    warm_up_iters = model_config['warm_up_iters']


    logging.info(f'transformer is set up with {num_iterations}')

    for batch_idx, (data, target, seq_len, _) in enumerate(data_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            my_stream.synchronize()
            sync_info.pre_measurement_prep(tid)
            # start timer
            start_time = time.time()

        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                # with torch.cuda.amp.autocast(enable_autocast):
                loss, mem = model(data, target, mem)
                loss = loss.float().mean().type_as(loss)

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            with torch.cuda.stream(my_stream):
                # if model_config['use_fp16']:
                #     if model_config['amp'] == 'pytorch':
                #         scaler.scale(loss).backward()
                #         scaler.unscale_(optimizer)
                #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                #     elif model_config['amp'] == 'apex':
                #         with amp.scale_loss(loss, optimizer, delay_unscale=False) as scaled_loss:
                #             scaled_loss.backward()
                #         torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip)
                # else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                # if model_config['use_fp16'] and model_config['amp'] == 'pytorch':
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                optimizer.step()
                model.zero_grad()

        if batch_idx == num_iterations - 1:
            # reached the last iteration
            break
    my_stream.synchronize()
    duration = time.time() - start_time
    sync_info.post_measurement_prep(tid)
    sync_info.write_kv(f'duration{tid}', duration)
    logging.info(f'tid {tid} it takes {duration} seconds to train transformer')
    return duration



