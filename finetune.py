import os
import fire
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from transformers import CTRLLMHeadModel, CTRLTokenizer, AdamW, WarmupLinearSchedule

from dataset import TextDataset
from model import DummyModel

def finetune(checkpoint="ctrl", train_path="./processed_dataset.pkl", save_dir='./checkpoints', learning_rate=1e-5, batch_size=4, epochs=2, gradient_accumulation_steps=1, logging_steps=10, histogram_steps=100, accelerator='GPU', subset=False):

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

    writer = SummaryWriter()

    writer.add_text("Checkpoint", checkpoint, 0)
    writer.add_text("Learning rate", str(learning_rate), 0)
    writer.add_text("Batch size", str(batch_size), 0)
    writer.add_text("Epochs", str(epochs), 0)
    writer.add_text("Gradient accumulation steps", str(gradient_accumulation_steps), 0)
    writer.add_text("Histogram interval", str(histogram_steps), 0)

    train_dataset = TextDataset(train_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if subset:
        model = DummyModel().to(device)
    else:
        model = CTRLLMHeadModel.from_pretrained(checkpoint).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    train_steps = int(len(train_dataloader) / gradient_accumulation_steps * epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(0.1 * train_steps), t_total=train_steps)

    # if accelerator == 'GPU':
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / batch_size / gradient_accumulation_steps)):
            inputs, labels = batch.to(device), batch.to(device)

            out = model(inputs, labels=labels)
            loss, out = out[:2]

            loss = loss / gradient_accumulation_steps

            # if accelerator == 'GPU':
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                if global_step % logging_steps == 0:
                    writer.add_scalar("train_loss", loss.item(), global_step)
                    writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)

                if global_step % histogram_steps == 0:
                    try:
                        for name, param in model.named_parameters():
                            writer.add_histogram(f'{name}', param, global_step, bins='sqrt')
                            if param.grad is not None:
                                writer.add_histogram(f'{name}.grad', param.grad, global_step, bins='sqrt')
                    except:
                        print('Error logging histograms')

                # if accelerator == 'GPU':
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                # else:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)


                if accelerator == 'TPU':
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
    
                scheduler.step()

                optimizer.zero_grad()

                global_step += 1

        model.to('cpu')
        model.save_pretrained(save_dir)        

if __name__ == "__main__":
    fire.Fire()
