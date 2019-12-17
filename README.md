# lm-experiments

Some code adapted from https://github.com/huggingface/transformers and https://github.com/salesforce/ctrl

IMDB dataset from: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

-   tpu stuff works; seems like each core has 64/8=8gb of ram; only 117M with a batch size of 8 works in about 20m/epoch

# ToDo

-   profile code
-   multi gpu/machine training
-   tpu training
-   pplm
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

# Acknowledgements

some code taken from huggingface/transformers
