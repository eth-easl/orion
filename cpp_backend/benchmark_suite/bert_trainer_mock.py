import torch
import threading
import time
import modeling

from optimization import BertAdam


def bert_loop(batchsize, train, local_rank, barriers, tid):

    print("ENTER!")

    barriers[0].wait()
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
        "vocab_size": 30528
    }

    config = modeling.BertConfig.from_dict(model_config)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    print("-------------- thread id:  ", threading.get_native_id())

    input_ids = torch.ones((batchsize, 384)).to(torch.int64).to(0)
    segment_ids = torch.ones((batchsize, 384)).to(torch.int64).to(0)
    input_mask = torch.ones((batchsize, 384)).to(torch.int64).to(0)
    start_positions = torch.zeros((batchsize)).to(torch.int64).to(0)
    end_positions = torch.ones((batchsize)).to(torch.int64).to(0)

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

    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        batch_idx = 0
        torch.cuda.synchronize()
                                
        while batch_idx < 30:
    
            print(f"submit!, batch_idx is {batch_idx}")
            #torch.cuda.profiler.cudart().cudaProfilerStart()

            if train:
                optimizer.zero_grad()
                start_logits, end_logits = model(input_ids, segment_ids, input_mask)
                ignored_index = start_logits.size(1)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    output = model(input_ids, segment_ids, input_mask)

            #torch.cuda.profiler.cudart().cudaProfilerStop()
            #print(output)
            batch_idx += 1

            print("sent everything!")

            #if (batch_idx == 1) and train: # for backward
            #    barriers[0].wait()

            if batch_idx < 30:
                barriers[0].wait()

    print("Epoch took: ", time.time()-start)
