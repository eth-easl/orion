# TICK-TOCK scheduling

This directory contains a basic implementation of TICK-TOCK scheduling using Python threads, and torch.cuda streams and events.
It is based on the description provided in [WAVELET: EFFICIENT DNN TRAINING WITH TICK-TOCK SCHEDULING (MLSys'21)](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf).

What would be an interesting next step is implementing the memory management support described in [Zico: Efficient GPU Memory Sharing for
Concurrent DNN Training (ATC'21)](https://www.usenix.org/system/files/atc21-lim.pdf).

## Supported Models

### BERT

It is mainly adapted from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
(hereinafter referenced as "source") with the following difference:
1. No distributed training is available. Training is only on one GPU. (same for all other models)
2. For the lack of C++ enabled `apex` (which will be installed on Google Cloud soon), 
several places have been commented and marked with `TODO`.
3. I didn't add these two settings as they are not found in other training scripts:
```python
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
```
4. Pretrained checkpoint is not loaded.

`train_bert_on_squad.py` serves as the entry point to fine tune bert on SQUAD dataset.


#### Confusions
The original script seems to indicate the tokenizer only works
for bert large model:
```python
tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
```
However as I digged out it should work for both.

### dcgan

Mainly copied from https://github.com/pytorch/examples/blob/main/dcgan/main.py with no difference.

### gnmt
The differences with https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT
are:
1. We don't preallocate space (e.g. run a forward and backward iteration without updating weights) before starting a new epoch.

### retinanet
Differences with https://github.com/mlcommons/training/tree/master/single_stage_detector:
1. They passed the `pin_memory` as `True` which does not appear in other models.
2. I didn't set up the learning rate scheduler.
3. For auto mixed precision they only use pytorch provided version, but not apex.

### nasnet and vision
No difference

### transformer
Differences with https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL.
1. They also explicitly disable profiling by the following, which I didn't do.
```python
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
```
2. They also do the following which is not done in other models:
```python
if 'apex' in sys.modules:
    amp.register_half_function(torch, 'einsum')
```
3. I didn't add the learning rate scheduler.
4. Since the default `batch_chunk` is 1, which is used to split each batch and train with gradient accumulation,
I omitted a lot of code to split data and train separately.

### miscellaneous
Some models use a seed for reproducibility, where others don't. 