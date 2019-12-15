import os
import random
import fire
import glob
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from transformers import GPT2Tokenizer


def imdb(path, save_dir):
    imdb = pd.read_csv(path)

    reviews = imdb['review'].values

    with open(os.path.join(save_dir, 'imdb.txt'), 'w') as f:
        for i, review in tqdm(enumerate(reviews), total=len(reviews)):
            f.write(review)


def preprocess(data_folder, save_path, name, checkpoint="gpt2", seq_len=256, subset=False, control_code=None):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

    batches = []
    paths = glob.glob(f"{data_folder}/*.txt")

    for path in tqdm(paths, total=len(paths)):
        with open(path, encoding="utf-8") as file:
            text = file.read()

        if subset:
            text = text[:10000]

        # Remove extra spaces that cause errors when tokenizing
        text = " ".join(text.split())

        if checkpoint == 'gpt2':
            text = "<|endoftext|> " + text
        elif checkpoint == 'ctrl':
            text = control_code + " " + text

        tokenized_text = tokenizer.encode(text)

        for i in range(0, len(tokenized_text) - seq_len + 1, seq_len):
            batches.append(tokenized_text[i: i + seq_len])

    random.shuffle(batches)

    with open(os.path.join(save_path, f'{name}_data.pkl'), "wb") as handle:
        pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire()
