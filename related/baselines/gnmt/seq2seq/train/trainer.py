# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import time
from itertools import cycle

import numpy as np
import torch
import torch.optim
import torch.utils.data
from apex.parallel import DistributedDataParallel
from apex import amp

from gnmt.seq2seq.train.fp_optimizers import FP16Optimizer
from gnmt.seq2seq.train.fp_optimizers import FP32Optimizer
from gnmt.seq2seq.train.fp_optimizers import AMPOptimizer
from gnmt.seq2seq.train.lr_scheduler import WarmupMultiStepLR
from gnmt.seq2seq.utils import AverageMeter
from gnmt.seq2seq.utils import sync_workers

from utils.sync_control import *


class Seq2SeqTrainer:
    """
    Seq2SeqTrainer
    """

    def __init__(self,
                 model,
                 criterion,
                 opt_config,
                 scheduler_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 save_info={},
                 save_dir='.',
                 train_iterations=0,
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 loss_scaling={},
                 intra_epoch_eval=0,
                 prealloc_mode='always',
                 warmup=0,
                 iter_size=1,
                 translator=None,
                 verbose=False):
        """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param opt_config: dictionary with options for the optimizer
        :param scheduler_config: dictionary with options for the learning rate
            scheduler
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_freq: saves checkpoint every 'save_freq' iterations
        :param grad_clip: coefficient for gradient clipping
        :param save_info: dict with additional state stored in each checkpoint
        :param save_dir: path to the directiory for checkpoints
        :param train_iterations: total number of training iterations to execute
        :param checkpoint_filename: name of files with checkpoints
        :param keep_checkpoints: max number of checkpoints to keep
        :param math: arithmetic type
        :param loss_scaling: options for dynamic loss scaling
        :param intra_epoch_eval: number of additional eval runs within each
            training epoch
        :param prealloc_mode: controls preallocation,
            choices=['off', 'once', 'always']
        :param warmup: number of warmup iterations for performance counters
        :param iter_size: number of iterations between weight updates
        :param translator: instance of Translator, runs inference on test set
        :param verbose: enables verbose logging
        """
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.device = next(model.parameters()).device
        self.print_freq = print_freq
        self.verbose = verbose
        self.loss = None
        self.translator = translator
        self.intra_epoch_eval = intra_epoch_eval
        self.warmup = warmup
        self.iter_size = iter_size
        self.prealloc_mode = prealloc_mode
        self.preallocated = False

        self.distributed = torch.distributed.is_initialized()
        self.batch_first = model.batch_first

        params = self.model.parameters()

        if math == 'manual_fp16':
            self.fp_optimizer = FP16Optimizer(
                self.model, grad_clip,
                loss_scale=loss_scaling['init_scale'],
                dls_upscale_interval=loss_scaling['upscale_interval']
            )
            params = self.fp_optimizer.fp32_params
        elif math == 'fp32' or math == 'tf32':
            self.fp_optimizer = FP32Optimizer(self.model, grad_clip)

        opt_name = opt_config.pop('optimizer')
        self.optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
        logging.info(f'Using optimizer: {self.optimizer}')

        self.scheduler = WarmupMultiStepLR(self.optimizer, train_iterations,
                                           **scheduler_config)

        if math == 'fp16':
            self.model, self.optimizer = amp.initialize(
                self.model,
                self.optimizer,
                cast_model_outputs=torch.float16,
                keep_batchnorm_fp32=False,
                opt_level='O2')

            self.fp_optimizer = AMPOptimizer(
                self.model,
                grad_clip,
                loss_scale=loss_scaling['init_scale'],
                dls_upscale_interval=loss_scaling['upscale_interval']
            )

        if self.distributed:
            self.model = DistributedDataParallel(self.model)

    def iterate(self, src, tgt, thread_id: int, sync_info, my_stream, batch_idx):
        src, src_length = src
        tgt, tgt_length = tgt
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)
        with ForwardControl(thread_id=thread_id, sync_info=sync_info, batch_idx=batch_idx, stream=my_stream):
            with torch.cuda.stream(my_stream):
                # if self.batch_first:
                #     output = self.model(src, src_length, tgt[:, :-1])
                #     tgt_labels = tgt[:, 1:]
                #     T, B = output.size(1), output.size(0)
                # else:
                output = self.model(src, src_length, tgt[:-1])
                tgt_labels = tgt[1:]
                T, B = output.size(0), output.size(1)

        with BackwardControl(thread_id=thread_id, sync_info=sync_info, batch_idx=batch_idx, stream=my_stream):
            with torch.cuda.stream(my_stream):
                loss = self.criterion(output.view(T * B, -1),
                                      tgt_labels.contiguous().view(-1))
                loss_per_batch = loss.item()
                loss /= (B * self.iter_size)
                # .step() includes zero_grad
                self.fp_optimizer.step(loss, self.optimizer, self.scheduler, update=True)

        return loss_per_batch

