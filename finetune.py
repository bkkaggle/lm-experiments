import fire
import glob
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import CTRLLMHeadModel, CTRLTokenizer

def preprocess(
    path,
    control_code,
    save_path="processed_dataset.pkl",
    checkpoint="ctrl",
    seq_len=256,
    subset=False,
):
    tokenizer = CTRLTokenizer.from_pretrained(checkpoint)

    examples = []
    paths = glob.glob(f"{path}/*.txt")

    for path in paths:
        with open(path, encoding='utf-8') as file:
            text = file.read()

        if subset:
            text = text[:10000]

        # Remove extra spaces that cause errors when tokenizing
        text = " ".join(text.split())

        tokenized_text = tokenizer.encode(text)

        batches = []
        for i in tqdm(
            range(0, len(tokenized_text) - seq_len + 1, seq_len),
            total=int(len(tokenized_text) / seq_len),
        ):
            batches.append(
                tokenizer.encode(control_code) + tokenized_text[i : i + seq_len]
            )
            if i == 5:
                break

        with open(save_path, "wb") as handle:
            pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire()
