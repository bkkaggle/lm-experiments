import os
import fire
import glob
import pickle

import numpy as np
from tqdm import tqdm

import torch

from transformers import GPT2Tokenizer

def preprocess(data_folder, save_path, name="moby", checkpoint="gpt2", seq_len=256, val_size=0.1, subset=False):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

    batches = []
    paths = glob.glob(f"{data_folder}/*.txt")

    for path in paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()

        if subset:
            text = text[:10000]

        # Remove extra spaces that cause errors when tokenizing
        text = " ".join(text.split())

        tokenized_text = tokenizer.encode(text)

        for i in tqdm(range(0, len(tokenized_text) - seq_len + 1, seq_len), total=int(len(tokenized_text) / seq_len),):
            batches.append(tokenized_text[i : i + seq_len])

    train_len = int(len(batches) * (1 - val_size))
    train_batches = batches[:train_len]
    val_batches = batches[train_len:]

    with open(os.path.join(save_path, f'{name}_data_train.pkl'), "wb") as handle:
        pickle.dump(train_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, f'{name}_data_val.pkl'), "wb") as handle:
        pickle.dump(val_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    fire.Fire(preprocess)