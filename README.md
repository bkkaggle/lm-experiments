# lm-experiments

Some code adapted from https://github.com/huggingface/transformers and https://github.com/salesforce/ctrl

IMDB dataset from: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

-   tpu stuff works; seems like each core has 64/8=8gb of ram; only 117M with a batch size of 8 works in about 20m/epoch

-   work with gpt2 or ctrl automatically

-   on the fly tokenization
-   take out multiple datasets
-   multi gpu/machine training
-   saving and resuming

# ToDo

# Done

-   use config files
