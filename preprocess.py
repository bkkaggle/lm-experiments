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

    for i, review in tqdm(enumerate(reviews), total=len(reviews)):
        with open(os.path.join(save_dir, f'review_{i}.txt'), 'w') as f:
            f.write(review)

def preprocess(data_folder, save_path, name, checkpoint="gpt2", seq_len=256, val_size=0.1, subset=False):
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

        tokenized_text = tokenizer.encode(text)

        for i in range(0, len(tokenized_text) - seq_len + 1, seq_len):
            batches.append(tokenized_text[i : i + seq_len])

    random.shuffle(batches)

    train_len = int(len(batches) * (1 - val_size))
    train_batches = batches[:train_len]
    val_batches = batches[train_len:]

    with open(os.path.join(save_path, f'{name}_data_train.pkl'), "wb") as handle:
        pickle.dump(train_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, f'{name}_data_val.pkl'), "wb") as handle:
        pickle.dump(val_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    fire.Fire()