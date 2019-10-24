import fire
import pickle
import mlcrate as mlc

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from transformers import CTRLLMHeadModel, CTRLTokenizer, AdamW, WarmupLinearSchedule

from dataset import TextDataset
from model import DummyModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def finetune(
    checkpoint="ctrl",
    train_path="./processed_dataset.pkl",
    learning_rate=1e-5,
    batch_size=4,
    epochs=2,
    gradient_accumulation_steps=1,
    histogram_interval=100,
    fp16=False,
    subset=False,
):
    timer = mlc.time.Timer()
    writer = SummaryWriter()

    writer.add_text("Checkpoint", checkpoint, 0)
    writer.add_text("Learning rate", str(learning_rate), 0)
    writer.add_text("Batch size", str(batch_size), 0)
    writer.add_text("Epochs", str(epochs), 0)
    writer.add_text("Gradient accumulation steps", str(gradient_accumulation_steps), 0)
    writer.add_text("Histogram interval", str(histogram_interval), 0)

    train_dataset = TextDataset(train_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    if subset:
        model = DummyModel().to(device)
    else:
        model = CTRLLMHeadModel.from_pretrained(checkpoint).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    train_steps = int(len(train_dataloader) / gradient_accumulation_steps * epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=int(0.1 * train_steps), t_total=train_steps
    )

    if fp16 == True:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        timer.add(epoch)

        train_loss = 0

        model.train()
        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=(int(len(train_dataset) / batch_size / gradient_accumulation_steps)),
        ):
            inputs, labels = batch.to(device), batch.to(device)

            out = model(inputs, labels=labels)
            loss, out = out[:2]


if __name__ == "__main__":
    fire.Fire()
