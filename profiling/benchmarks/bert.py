import torch
import threading
import time
import modeling

from optimization import BertAdam

def bert(batchsize, local_rank, do_eval=True, profile=True):

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


    input_ids = torch.ones((batchsize, 384)).to(torch.int64).to(0)
    segment_ids = torch.ones((batchsize, 384)).to(torch.int64).to(0)
    input_mask = torch.ones((batchsize, 384)).to(torch.int64).to(0)
    start_positions = torch.zeros((batchsize)).to(torch.int64).to(0)
    end_positions = torch.ones((batchsize)).to(torch.int64).to(0)


    model = modeling.BertForQuestionAnswering(config).to(0)

    if do_eval:
        model.eval()
    else:
        model.train()
        param_optimizer = list(model.named_parameters())

        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=100)

    batch_idx = 0
    torch.cuda.synchronize()

    while batch_idx < 1:

        if batch_idx == 0:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()

        if do_eval:
            with torch.no_grad():
                output = model(input_ids, segment_ids, input_mask)
        else:
            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)
            ignored_index = start_logits.size(1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            loss.backward()
            optimizer.step()

        if batch_idx == 0:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()

        batch_idx += 1

    print("Done!")

if __name__ == "__main__":
    bert(8, 0,False, 'nsys')
