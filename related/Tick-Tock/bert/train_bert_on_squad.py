import pickle
import time

from apex import amp
from apex.optimizers import FusedAdam
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import random
import numpy as np
from bert.schedulers import LinearWarmUpScheduler
from bert.optimization import BertAdam
import bert.modeling as modeling
from bert.squad_example import *
from bert.tokenization import BertTokenizer
import os
import utils.constants as constants
from utils.sync_info import SyncInfo
from utils.sync_control import *


from apex.multi_tensor_apply import multi_tensor_applier
# TODO: uncomment to enable fp16
# class GradientClipper:
#     """
#     Clips gradient norm of an iterable of parameters.
#     """
#     def __init__(self, max_grad_norm):
#         self.max_norm = max_grad_norm
#         if multi_tensor_applier.available:
#             import amp_C
#             self._overflow_buf = torch.cuda.IntTensor([0])
#             self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
#             self.multi_tensor_scale = amp_C.multi_tensor_scale
#         else:
#             raise RuntimeError('Gradient clipping requires cuda extensions')
#
#     def step(self, parameters):
#         l = [p.grad for p in parameters if p.grad is not None]
#         total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
#         total_norm = total_norm.item()
#         if (total_norm == float('inf')): return
#         clip_coef = self.max_norm / (total_norm + 1e-6)
#         if clip_coef < 1:
#             multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)


def setup_model(model_config):
    arch = model_config['arch']
    config_file = os.path.join(
        model_config['large_model_dir'] if arch == 'large' else model_config['base_model_dir'],
        'bert_config.json'
    )
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForQuestionAnswering(config)
    # TODO: load the pre-trained weights into the model
    return model



def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    squad_version = model_config['squad_version']
    squad_file = constants.squad_version1 if squad_version == 1 else constants.squad_version2
    train_examples = read_squad_examples(
        input_file=squad_file,
        is_training=True,
        version_2_with_negative=squad_version == 2
    )
    batch_size = model_config['batch_size']
    num_train_optimization_steps = int(len(train_examples) / batch_size) * num_epochs
    model = setup_model(model_config)
    model = model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if model_config['use_fp16']:
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=5e-5, bias_correction=False)
        loss_scale = model_config['fp16_loss_scale']
        if loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=loss_scale)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=0.1,
                                          total_steps=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5,
                             warmup=0.1,
                             t_total=num_train_optimization_steps)
    vocab_file = os.path.join(
        model_config['large_model_dir'] if model_config['arch'] == 'large' else model_config['base_model_dir'],
        'vocab.txt'
    )
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True, max_len=512)

    cache_features_file = os.path.join(os.path.dirname(squad_file), 'cache_features')
    try:
        with open(cache_features_file, 'rb') as reader:
            train_features = pickle.load(reader)
    except:
        print(f'no cache file detected as {cache_features_file}; building features from squad examples...')
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=True
        )
        print(f'saving features to {cache_features_file}')
        with open(cache_features_file, 'wb') as writer:
            pickle.dump(train_features, writer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    model.train()
    # TODO: this requires amp_C package which isn't avaiable for a pure python apex
    # gradClipper = GradientClipper(max_grad_norm=1.0)
    with TrainingControl(sync_info=sync_info, device=device), torch.cuda.stream(my_stream):
        for _ in range(num_epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                with ForwardControl(tid, batch_idx, sync_info, my_stream):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    start_logits, end_logits = model(input_ids, segment_ids, input_mask)

                with BackwardControl(tid, batch_idx, sync_info, my_stream):
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = start_logits.size(1)
                    start_positions.clamp_(0, ignored_index)
                    end_positions.clamp_(0, ignored_index)
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2
                    if model_config['use_fp16']:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    # gradient clipping
                    # TODO: uncomment to enable amp_C
                    # gradClipper.step(amp.master_params(optimizer))
                    if model_config['use_fp16']:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()


