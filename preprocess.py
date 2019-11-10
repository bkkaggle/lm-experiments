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

def starwars(path, save_dir):
    paths = glob.glob(f"{path}/*.txt")

    i = 0
    for path in paths:        
        # From: https://www.kaggle.com/xvivancos/star-wars-movie-scripts/discussion/58113
        starwars = pd.read_table(path, delim_whitespace=True, header=0, escapechar='\\')

        lines = starwars['dialogue'].values

        script = ''
        for _, line in tqdm(enumerate(lines), total=len(lines)):
            script += ' ' + line

        with open(os.path.join(save_dir, f'starwars_{i}.txt'), 'w') as f:
            f.write(script)

        i += 1

def preprocess(data_folder, save_path, name, checkpoint="gpt2", seq_len=256, subset=False):
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

    with open(os.path.join(save_path, f'{name}_data.pkl'), "wb") as handle:
        pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    fire.Fire()
