import torch
import threading
import time
import modeling
import numpy as np
import json

from optimization import BertAdam

from ctypes import *
import os

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.input_ids = torch.ones((self.batchsize, 384), pin_memory=True).to(torch.int64)
        self.segment_ids = torch.ones((self.batchsize, 384), pin_memory=True).to(torch.int64)
        self.input_mask = torch.ones((self.batchsize, 384), pin_memory=True).to(torch.int64)
        self.start_positions = torch.zeros((self.batchsize,), pin_memory=True).to(torch.int64)
        self.end_positions = torch.ones((self.batchsize,), pin_memory=True).to(torch.int64)


    def __iter__(self):
        return self

    def __next__(self):
        return self.input_ids, self.segment_ids, self.input_mask, self.start_positions, self.end_positions


def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)


def check_stop(backend_lib):
    return backend_lib.stop()

def bert_loop(batchsize, train, num_iters, rps, uniform, dummy_data, local_rank, barriers, client_barrier, tid):

    seed_everything(42)
    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/orion/src/cuda_capture/libinttemp.so")

    if rps > 0:
        if uniform:
            sleep_times = [1/rps]*num_iters
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    barriers[0].wait()
    
    if (train and tid==1):
        time.sleep(5)
        

    if (not train):
        model_config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "output_all_encoded_layers": False,
            "type_vocab_size": 2,
            "vocab_size": 30522
        }
    else:
        model_config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30522
        }

    config = modeling.BertConfig.from_dict(model_config)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    print("-------------- thread id:  ", threading.get_native_id())


    model = modeling.BertForQuestionAnswering(config).to(0)

    if train:
        model.train()
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=100)
    else:
        model.eval()

    train_loader = DummyDataLoader(batchsize)
    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)
    
    #  open loop
    timings = []
    next_startup = time.time()
    open_loop = True


    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
                                
        while batch_idx < num_iters:
    
            if train:
                print(f"Start iter {batch_idx}")
                optimizer.zero_grad()
                input_ids, segment_ids, input_mask, start_positions, end_positions = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank), batch[3].to(local_rank), batch[4].to(local_rank)
                start_logits, end_logits = model(input_ids, segment_ids, input_mask)
                ignored_index = start_logits.size(1)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                loss.backward()
                optimizer.step()
                block(backend_lib, batch_idx)
                batch_idx, batch = next(train_iter)
                if (batch_idx == 1): # for backward
                    barriers[0].wait()
                if batch_idx == 10:
                    barriers[0].wait()
                    start = time.time()
                if check_stop(backend_lib):
                    print("---- STOP!")
                    break
                if batch_idx==290:
                    torch.cuda.profiler.cudart().cudaProfilerStart()
            else:
                with torch.no_grad():
                    cur_time = time.time()
                    #### OPEN LOOP ####
                    if open_loop:
                        if (cur_time >= next_startup):
                            print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                            if batch_idx==50:
                                torch.cuda.profiler.cudart().cudaProfilerStart()
                            input_ids, segment_ids, input_mask = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                            output = model(input_ids, segment_ids, input_mask)
                            block(backend_lib, batch_idx)
                            req_time = time.time()-next_startup
                            timings.append(req_time)
                            print(f"Client {tid} finished! Wait! It took {req_time}")
                            if batch_idx>=10:
                                next_startup += sleep_times[batch_idx]
                            else:
                                next_startup = time.time()
                            batch_idx,batch = next(train_iter)
                            if ((batch_idx == 1) or (batch_idx == 10)):
                                barriers[0].wait()
                                if (batch_idx==10):
                                    #time.sleep(1)
                                    next_startup = time.time()
                                    start = time.time()
                            dur = next_startup-time.time()
                            if (dur>0):
                                time.sleep(dur)
                            if check_stop(backend_lib):
                                print("---- STOP!")
                                break

                    else:
                        ### CLOSED LOOP ###
                        print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                        input_ids, segment_ids, input_mask = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                        output = model(input_ids, segment_ids, input_mask)
                        print(f"Client {tid} finished! Wait!")
                        if ((batch_idx == 1) or (batch_idx == 10)):
                            barriers[0].wait()
                        batch_idx,batch = next(train_iter)


    torch.cuda.profiler.cudart().cudaProfilerStop()
    barriers[0].wait()
    total_time = time.time() - start


    if not train and len(timings)>50:
        timings = timings[50:]
        timings = sorted(timings)
        p50 = np.percentile(timings, 50) * 1000
        p95 = np.percentile(timings, 95) * 1000
        p99 = np.percentile(timings, 99) * 1000
        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")

        data = {
            'p50_latency': p50,
            'p95_latency': p95,
            'p99_latency': p99,
            'throughput': (batch_idx-10)/total_time
        }
    else:
        data = {
            'throughput': (batch_idx-10)/total_time
        }
    with open(f'client_{tid}.json', 'w') as f:
        json.dump(data, f)

    print("Finished! Ready to join!")
