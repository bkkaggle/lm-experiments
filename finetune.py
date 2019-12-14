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

from dataset import TextDataset
from model import DummyModel
from sample import sample

from config import Config

import wandb
wandb.init(project="transformer-experiments")


def finetune(**kwargs):
    config = Config(**kwargs)

    if config.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

    elif config.accelerator == 'GPU':
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        from apex import amp

    elif config.accelerator == 'CPU':
        device = torch.device("cpu")

    train_dataset = TextDataset(config.dataset_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    if config.subset:
        model = DummyModel().to(device)
    else:
        model = GPT2LMHeadModel.from_pretrained(config.checkpoint).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(config.checkpoint)

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(
    #         nd in n for nd in no_decay)], "weight_decay": 0.0},
    #     {"params": [p for n, p in model.named_parameters() if any(
    #         nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]

    # train_steps = int(len(train_dataloader) /
    #                   config.gradient_accumulation_steps * config.epochs)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
    #     0.1 * train_steps), num_training_steps=train_steps)

    # if config.accelerator == 'GPU':
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level="O1", loss_scale="dynamic")

    # wandb.watch(model, log='parameters')

    # global_step = 0
    # gradients = {}

    # for epoch in range(config.epochs):
    #     train_loss = 0

    #     print(f"Epoch: {epoch}")

    #     model.train()
    #     for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / config.batch_size)):
    #         inputs, labels = batch.to(device), batch.to(device)

    #         out = model(inputs, labels=labels)
    #         loss = out[0]

    #         loss = loss / config.gradient_accumulation_steps

    #         train_loss += loss.item()

    #         if config.accelerator == 'GPU':
    #             with amp.scale_loss(loss, optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             loss.backward()

    #         if (i + 1) % config.gradient_accumulation_steps == 0:
    #             if config.accelerator == 'GPU':
    #                 torch.nn.utils.clip_grad_norm_(
    #                     amp.master_params(optimizer), 1)
    #             else:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    #             if config.accelerator == 'TPU':
    #                 xm.optimizer_step(optimizer, barrier=True)
    #             else:
    #                 optimizer.step()

    #             scheduler.step()

    #             if global_step % config.logging_steps == 0:
    #                 wandb.log({"train_loss": loss.item() * config.gradient_accumulation_steps,
    #                            "learning_rate": scheduler.get_lr()[0]}, step=global_step)

    #                 if global_step % config.histogram_steps == 0:
    #                     for name, param in model.named_parameters():
    #                         if param.grad is not None:
    #                             try:
    #                                 gradients[f"gradients/{name}"] = wandb.Histogram(
    #                                     param.grad.detach().cpu().numpy())
    #                             except:
    #                                 pass

    #                 wandb.log(gradients, step=global_step)

    #             optimizer.zero_grad()

    #             global_step += 1

    #     train_loss /= (i + 1)
    #     train_loss *= config.gradient_accumulation_steps
    #     train_perplexity = torch.exp(torch.tensor(train_loss))

    #     wandb.log({"train_epoch_loss": train_loss,
    #                "train_epoch_perplexity": train_perplexity}, step=global_step)

    #     message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity}'
    #     print(message)

    #     model.to('cpu')
    #     model.save_pretrained(config.save_dir)
    #     tokenizer.save_pretrained(config.save_dir)

    #     model.to(device)

    print('Sampling from model:\n')
    for _ in range(config.n_samples):
        out = sample(" ", model, tokenizer, length=config.sample_len, temperature=config.temperature,
                     top_k=config.top_k, top_p=config.top_p, repetition_penalty=config.repetition_penalty)
        print(out)
        print('\n')


if __name__ == "__main__":
    fire.Fire(finetune)
