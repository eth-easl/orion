# TICK-TOCK scheduling

This directory contains a basic implementation of TICK-TOCK scheduling using Python threads, and torch.cuda streams and events.
It is based on the description provided in [WAVELET: EFFICIENT DNN TRAINING WITH TICK-TOCK SCHEDULING (MLSys'21)](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf).

What would be an interesting next step is implementing the memory management support described in [Zico: Efficient GPU Memory Sharing for
Concurrent DNN Training (ATC'21)](https://www.usenix.org/system/files/atc21-lim.pdf).

### Supported Models
