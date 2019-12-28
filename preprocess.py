import os
import random
import fire
import glob
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from transformers import GPT2TokenizerFast, CTRLTokenizer

TOKENIZER_CLASSES = {
    'gpt2': GPT2TokenizerFast,
    'ctrl': CTRLTokenizer
}


def imdb(path, save_dir,  model_type='gpt2', control_code=None):
    imdb = pd.read_csv(path)

    reviews = imdb['review'].values
    print(f'There are {len(reviews)} reviews')

    with open(os.path.join(save_dir, 'imdb.txt'), 'w') as f:
        for i, review in tqdm(enumerate(reviews), total=len(reviews)):

            if model_type == 'gpt2':
                review = "<|endoftext|> " + review
            elif model_type == 'ctrl':
                review = control_code + " " + review

            f.write(review)


def all_the_news(path, save_dir, model_type='gpt2', checkpoint='gpt2', dataset_name=None, seq_len=256, train_split=0.9, control_code=None):
    tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained(checkpoint)

    df = pd.read_csv(path)

    articles = ""
    for article in tqdm(df['content'].values):
        if model_type == 'gpt2':
            article = "<|endoftext|> " + article
        elif model_type == 'ctrl':
            article = control_code + " " + article

        article = " ".join(article.split())

        articles = articles + " " + article

    start = time.time()
    articles = tokenizer.encode(articles)
    end = time.time()
    print(f'Time to tokenize: {end - start} seconds')

    batches = []
    for i in range(0, len(articles) - seq_len + 1, seq_len):
        batches.append(articles[i: i + seq_len])

    random.shuffle(batches)

    train_batches = batches[:int(len(batches) * train_split)]
    test_batches = batches[int(len(batches) * train_split):]

    if control_code:
        prefix = f'{dataset_name}-{checkpoint}-{control_code}'
    else:
        prefix = f'{dataset_name}-{checkpoint}'

    with open(os.path.join(save_dir, f'{prefix}-train.pkl'), "wb") as handle:
        pickle.dump(train_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, f'{prefix}-test.pkl'), "wb") as handle:
        pickle.dump(test_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire()
