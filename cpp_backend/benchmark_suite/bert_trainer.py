import torch
import threading
import time
import modeling

def bert_loop(batchsize, loader, local_rank, barrier, tid):

    barrier.wait()
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

    input_ids = torch.ones((8, 384)).to(torch.int64).to(0)
    segment_ids = torch.ones((8, 384)).to(torch.int64).to(0)
    input_mask = torch.ones((8, 384)).to(torch.int64).to(0)
    
    torch.cuda.profiler.cudart().cudaProfilerStart()
    model = modeling.BertForQuestionAnswering(config).to(0)

    model.eval()

    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        batch_idx = 0
        torch.cuda.synchronize()
                                

        while batch_idx < 1:
    
            print(f"submit!, batch_idx is {batch_idx}")
            #torch.cuda.profiler.cudart().cudaProfilerStart()

            with torch.no_grad():
                output = model(input_ids, segment_ids, input_mask)

            torch.cuda.profiler.cudart().cudaProfilerStop()
            #print(output)
            batch_idx += 1
            start_iter = time.time()                                                               
    print("Epoch took: ", time.time()-start)
