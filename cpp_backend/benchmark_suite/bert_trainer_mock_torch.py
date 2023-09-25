import torch
import threading
import time
import numpy as np
import modeling

from optimization import BertAdam

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.input_ids = torch.ones((self.batchsize, 384), pin_memory=False).to(torch.int64)
        self.segment_ids = torch.ones((self.batchsize, 384), pin_memory=False).to(torch.int64)
        self.input_mask = torch.ones((self.batchsize, 384), pin_memory=False).to(torch.int64)
        self.start_positions = torch.zeros((self.batchsize,), pin_memory=False).to(torch.int64)
        self.end_positions = torch.ones((self.batchsize,), pin_memory=False).to(torch.int64)

    def __iter__(self):
        return self

    def __next__(self):
        return self.input_ids, self.segment_ids, self.input_mask, self.start_positions, self.end_positions

def bert_loop(batchsize, train, num_iters, default, rps, uniform, dummy_data, local_rank, start_barriers, end_barriers, tid):

    start_barriers[0].wait()

    if rps > 0:
        if uniform:
            sleep_times = [1/rps]*num_iters
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    if default:
        s = torch.cuda.default_stream()
    else:
        s = torch.cuda.Stream()

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

    next_startup = time.time()
    open_loop = True
    timings = [0 for _ in range(num_iters)]

    with torch.cuda.stream(s):

        for i in range(1):
            print("Start epoch: ", i)


            while batch_idx < num_iters:

                start = time.time()

                if train:
                    start_iter = time.time()
                    #start_barriers[0].wait()
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
                    #s.synchronize()
                    print(f"Client {tid}, iter {batch_idx} took {time.time()-start_iter} sec")
                    batch_idx,batch = next(train_iter)
                    #end_barriers[0].wait()
                    if batch_idx == 10:
                        starttime = time.time()
                    if batch_idx == 300:
                        print(f"---------- Finished! total time is {time.time()-starttime}")
                else:
                    with torch.no_grad():
                        cur_time = time.time()
                        ###### OPEN LOOP #####
                        if (cur_time >= next_startup):
                            input_ids, segment_ids, input_mask = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                            output = model(input_ids, segment_ids, input_mask)
                            s.synchronize()
                            timings[batch_idx] = time.time()-next_startup
                            print(f"Client {tid}, Iteration {batch_idx} took {timings[batch_idx]} sec")
                            next_startup += sleep_times[batch_idx]
                            batch_idx,batch = next(train_iter)
                            if (batch_idx==10):
                                starttime = time.time()

    end_barriers[0].wait()

    if not train:
        timings = timings[2:]
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)

        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
        print(f"Total time is {time.time()-starttime} sec")
