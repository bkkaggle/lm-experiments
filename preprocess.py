import os
import random
import fire
import glob
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import wandb
from transformers import GPT2Tokenizer, CTRLTokenizer

from config import Config

wandb.init(project="transformer-experiments")

TOKENIZER_CLASSES = {
    'gpt2': GPT2Tokenizer,
    'ctrl': CTRLTokenizer
}


def imdb(path, save_dir):
    imdb = pd.read_csv(path)

    reviews = imdb['review'].values
    print(f'There are {len(reviews)} reviews')

    with open(os.path.join(save_dir, 'imdb.txt'), 'w') as f:
        for i, review in tqdm(enumerate(reviews), total=len(reviews)):
            f.write(review)


def preprocess(**kwargs):
    config = Config(**kwargs)

    tokenizer = TOKENIZER_CLASSES[config.model].from_pretrained(
        config.checkpoint)

    batches = []
    paths = glob.glob(f"{config.dataset_path}/*.txt")

    for path in tqdm(paths, total=len(paths)):
        with open(path, encoding="utf-8") as file:
            text = file.read()

        # Remove extra spaces that cause errors when tokenizing
        text = " ".join(text.split())

        if config.checkpoint == 'gpt2':
            text = "<|endoftext|> " + text
        elif config.checkpoint == 'ctrl':
            text = config.control_code + " " + text

        tokenized_text = tokenizer.encode(text)

        for i in range(0, len(tokenized_text) - config.seq_len + 1, config.seq_len):
            batches.append(tokenized_text[i: i + config.seq_len])

    random.shuffle(batches)

    if config.control_code:
        save_file = os.path.join(
            config.dataset_path, f'{config.dataset_name}-{config.checkpoint}-{config.control_code}.pkl')
    else:
        save_file = os.path.join(
            config.dataset_path, f'{config.dataset_name}-{config.checkpoint}.pkl')

    with open(save_file, "wb") as handle:
        pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire()
