import os
import fire
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from dataset import TextDataset, MultiDataset
from model import DummyModel
from sample import sample

# log examples; https://docs.wandb.com/library/python/log

import wandb
wandb.init(project="transformer-experiments")

def finetune(dataset_1_path, dataset_2_path=None, dataset_1_supersampling=1, checkpoint="gpt2", save_dir=wandb.run.dir, learning_rate=5e-5, batch_size=4, epochs=2, gradient_accumulation_steps=1, logging_steps=10, histogram_steps=100, accelerator='GPU', subset=False):

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

    if dataset_2_path:
        train_dataset = MultiDataset(dataset_1_path, dataset_2_path, dataset_1_supersampling)
    else:
        train_dataset = TextDataset(dataset_1_path)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'len dataset: {len(train_dataset)}')

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * train_steps), num_training_steps=train_steps)

    if accelerator == 'GPU':
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")

    wandb.watch(model, log='parameters')

    global_step = 0
    gradients = {}

    for epoch in range(epochs):
        train_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / batch_size)):
            inputs, labels = batch.to(device), batch.to(device)

            out = model(inputs, labels=labels)
            loss = out[0]

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

                if global_step % logging_steps == 0:
                    wandb.log({"train_loss": loss.item() * gradient_accumulation_steps, "learning_rate": scheduler.get_lr()[0]}, step=global_step)

                    if global_step & histogram_steps == 0:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                try:
                                    gradients[f"gradients/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())
                                except:
                                    pass
                    
                    wandb.log(gradients, step=global_step)

                optimizer.zero_grad()
    
                global_step += 1

        train_loss /= (i + 1) * gradient_accumulation_steps
        train_perplexity = torch.exp(torch.tensor(train_loss))

        wandb.log({"train_epoch_loss": train_loss, "train_epoch_perplexity": train_perplexity}, step=global_step)

        message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity}'
        print(message)

        print('Sampling from model:')
        sample(" ", model, tokenizer, length=256, temperature=1, top_p=0.9, repetition_penalty=1.2)

        model.to('cpu')
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        model.to(device)

if __name__ == "__main__":
    fire.Fire(finetune)
