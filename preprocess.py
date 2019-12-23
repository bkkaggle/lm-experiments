import os
import random
import fire
import glob
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from transformers import GPT2Tokenizer, CTRLTokenizer

TOKENIZER_CLASSES = {
    'gpt2': GPT2Tokenizer,
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


def all_the_news(path, save_dir, model_type='gpt2', control_code=None):
    df = pd.read_csv(path)

    articles = df['content'].values
    print(f'There are {len(articles)} articles')

    with open(os.path.join(save_dir, f'{model_type}.txt'), 'w') as f:
        for i, article in tqdm(enumerate(articles), total=len(articles)):

            if model_type == 'gpt2':
                article = "<|endoftext|> " + article
            elif model_type == 'ctrl':
                article = control_code + " " + article

            f.write(article)


def preprocess(dataset_path, model_type='gpt2', checkpoint='gpt2', dataset_name=None, seq_len=256, control_code=None):
    tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained(
        checkpoint)

    batches = []
    paths = glob.glob(f"{dataset_path}/*.txt")

    for path in tqdm(paths, total=len(paths)):
        with open(path, encoding="utf-8") as file:
            text = file.read()

        # Remove extra spaces that cause errors when tokenizing
        text = " ".join(text.split())

        tokenized_text = tokenizer.encode(text)

        for i in range(0, len(tokenized_text) - seq_len + 1, seq_len):
            batches.append(tokenized_text[i: i + seq_len])

    random.shuffle(batches)

    if control_code:
        save_file = os.path.join(
            dataset_path, f'{dataset_name}-{checkpoint}-{control_code}.pkl')
    else:
        save_file = os.path.join(
            dataset_path, f'{dataset_name}-{checkpoint}.pkl')

    with open(save_file, "wb") as handle:
        pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire()
