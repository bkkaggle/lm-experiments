# lm-experiments

Some code adapted from https://github.com/huggingface/transformers and https://github.com/salesforce/ctrl

IMDB dataset from: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

-   tpu stuff works; seems like each core has 64/8=8gb of ram; only 117M with a batch size of 8 works in about 20m/epoch

# ToDo

-   use wandb init
-   pplm
    -   distilgpt2
-   ctrl finetuning
-   t5

# Done

-   use config files
-   take out multiple datasets
-   work with gpt2 or ctrl automatically
-   save_steps
-   saving and resuming
-   use past
-   on the fly tokenization - wait for rust tokenizer, or just use the current one
    -   special tokens
    -   update other files
-   distilgpt2
-   named tensors - too experimental for now
-   batch generation
-   hooks for activations - not worth it
-   profile code

    -   kernprof -l -v (file)
    -   python -m line_profiler (file).lprof

-   tpu training
-   https://github.com/pytorch-tpu
-   multi gpu/machine training - later

# Benchmarks

## TPU Single core

-   maximum batch size: 16
-   by default, xla converts some layers to bfloat16 automatically
-   takes a few batches to get up to speed
-   it looks like the first training run in a session takes a lot longer
-   training is **: `moby, distilgpt2, seq_len 256, AdamW, batch size 4, grad steps 1: **`
-   more time is spent saving the model, so increase `save_steps`
-   mp doesn't work yet

| model      | seq len | optimizer | batch size | grad steps | bfloat16  | time                     | batches/s                                |
| ---------- | ------- | --------- | ---------- | ---------- | --------- | ------------------------ | ---------------------------------------- |
| distilgpt2 | 256     | adamw     | 16         | 1          | automatic | 2m, but startup was 1.5m | 3, if you don't include the startup time |

# GPU with apex 01

-   maximum 2^n batch size: 16

-   training is fast: `moby, distilgpt2, seq_len 256, AdamW, batch size 4, grad steps 4: 7.5 batches/second`

# Acknowledgements

some code taken from huggingface/transformers
