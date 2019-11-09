import os
import fire
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, WarmupLinearSchedule

from dataset import TextDataset
from model import DummyModel

# larger dataset

import wandb
wandb.init(project="transformer-experiments")

def finetune(train_path, checkpoint="gpt2", save_dir=wandb.run.dir, learning_rate=5e-5, batch_size=4, epochs=2, gradient_accumulation_steps=1, logging_steps=10, accelerator='GPU', subset=False):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if accelerator == 'TPU':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

        amp_mode = None

    elif accelerator == 'GPU':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from apex import amp

    elif accelerator == 'CPU':
        device = torch.device("cpu")

    wandb.config.checkpoint = checkpoint
    wandb.config.learning_rate = learning_rate
    wandb.config.batch_size = batch_size
    wandb.config.epochs = epochs
    wandb.config.gradient_accumulation_steps = gradient_accumulation_steps

    train_dataset = TextDataset(train_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if subset:
        model = DummyModel().to(device)
    else:
        model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    train_steps = int(len(train_dataloader) / gradient_accumulation_steps * epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(0.1 * train_steps), t_total=train_steps)

    if accelerator == 'GPU':
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    wandb.watch(model, log='all')

    global_step = 0

    for epoch in range(epochs):
        train_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / batch_size)):
            inputs, labels = batch.to(device), batch.to(device)

            out = model(inputs, labels=labels)
            loss = out[0]

            if global_step % logging_steps == 0:
                wandb.log({"train_loss": loss.item(), "learning_rate": scheduler.get_lr()[0]}, step=global_step)

            loss = loss / gradient_accumulation_steps

            train_loss += loss.item()

            if accelerator == 'GPU':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                if accelerator == 'GPU':
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                if accelerator == 'TPU':
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
    
                scheduler.step()

                optimizer.zero_grad()

                global_step += 1

        train_loss /= (i + 1)
        train_perplexity = torch.exp(torch.tensor(train_loss))

        wandb.log({"train_epoch_loss": train_loss, "train_epoch_perplexity": train_perplexity}, step=global_step)

        message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity}'
        print(message)

        model.to('cpu')
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        model.to(device)

if __name__ == "__main__":
    fire.Fire(finetune)
