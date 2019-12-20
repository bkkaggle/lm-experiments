import os
import fire
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2Tokenizer, CTRLTokenizer, AdamW, get_linear_schedule_with_warmup

from dataset import TextDataset
from model import DummyModel
from sample import sample

import wandb

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'test': (DummyModel(), GPT2Tokenizer)
}


# @profile
def finetune(dataset_path, save_dir, model_type, checkpoint, lr, batch_size, gradient_accumulation_steps, epochs, accelerator, logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug):
    wandb.init(project="transformer-experiments")

    if save_dir == None:
        save_dir = wandb.run.dir

    if debug:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    if accelerator == 'TPU':
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        device = xm.xla_device()

    elif accelerator == 'GPU':
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        from apex import amp

    elif accelerator == 'CPU':
        device = torch.device("cpu")

    train_dataset = TextDataset(dataset_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if accelerator == 'TPU':
        # from: https://github.com/pytorch/xla/issues/1191
        def len_parallelloader(self):
            return len(self._loader._loader)
        pl.PerDeviceLoader.__len__ = len_parallelloader

        train_dataloader = pl.ParallelLoader(
            train_dataloader, [device]).per_device_loader(device)

    model, tokenizer = MODEL_CLASSES[model_type]

    if model_type != 'test':
        model = model.from_pretrained(checkpoint).to(device)
    tokenizer = tokenizer.from_pretrained(checkpoint)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    train_steps = int(len(train_dataloader) /
                      gradient_accumulation_steps * epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        0.1 * train_steps), num_training_steps=train_steps)

    if os.path.exists(checkpoint):
        print('Loading optimizer and scheduler')

        optimizer.load_state_dict(torch.load(
            os.path.join(checkpoint, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(
            os.path.join(checkpoint, 'scheduler.pt')))

    if accelerator == 'GPU':
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", loss_scale="dynamic")

    wandb.watch(model, log='parameters')

    gradients = {}

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(checkpoint):
        global_step = int(checkpoint.split('-')[-1].split('/')[0])

        epochs_trained = global_step // (len(train_dataloader) //
                                         gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // gradient_accumulation_steps) * gradient_accumulation_steps

    for epoch in range(epochs_trained, epochs):
        train_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / batch_size)):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

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
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                if accelerator == 'TPU':
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                scheduler.step()

                if global_step % logging_steps == 0:
                    wandb.log({"train_loss": loss.item() * gradient_accumulation_steps,
                               "learning_rate": scheduler.get_lr()[0]}, step=global_step)

                    if global_step % histogram_steps == 0:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                try:
                                    gradients[f"gradients/{name}"] = wandb.Histogram(
                                        param.grad.detach().cpu().numpy())
                                except:
                                    pass

                    wandb.log(gradients, step=global_step)

                optimizer.zero_grad()

                global_step += 1

                # Must be in grad_accum block b/c if it is > 0, the model will get saved multiple times
                if global_step % save_steps == 0:
                    print(f'Saving model at global step: {global_step}')
                    checkpoint_dir = os.path.join(
                        save_dir, f'checkpoint-{global_step}')

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.save(optimizer.state_dict(), os.path.join(
                        checkpoint_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(
                        checkpoint_dir, 'scheduler.pt'))

        train_loss /= (i + 1)
        train_loss *= gradient_accumulation_steps
        train_perplexity = torch.exp(torch.tensor(train_loss))

        wandb.log({"train_epoch_loss": train_loss,
                   "train_epoch_perplexity": train_perplexity}, step=global_step)

        message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity}'
        print(message)

        print('Sampling from model:\n')
        sample(" ", model, tokenizer, length=sample_len, temperature=temperature,
               top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, n_samples=n_samples)
        print('\n')


def tpu(index, dataset_path, save_dir, model_type, checkpoint, lr, batch_size, gradient_accumulation_steps, epochs, accelerator, logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug):
    print(index)
    finetune(dataset_path, save_dir, model_type, checkpoint, lr, batch_size, gradient_accumulation_steps, epochs, accelerator,
             logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug)


def main(dataset_path='./data.pkl', save_dir=None, model_type='gpt2', checkpoint='distilgpt2', lr=5e-5, batch_size=4, gradient_accumulation_steps=1, epochs=1, accelerator='GPU', logging_steps=10, histogram_steps=100, save_steps=100, n_samples=1, sample_len=256, temperature=1, top_k=0, top_p=0, repetition_penalty=1, debug=False, n_cores=1):
    if accelerator == 'CPU' or accelerator == 'GPU':
        finetune(dataset_path, save_dir, model_type, checkpoint, lr, batch_size, gradient_accumulation_steps, epochs, accelerator,
                 logging_steps, histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug)
    else:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(tpu, args=(dataset_path, save_dir, model_type, checkpoint, lr, batch_size, gradient_accumulation_steps, epochs, accelerator, logging_steps,
                             histogram_steps, save_steps, n_samples, sample_len, temperature, top_k, top_p, repetition_penalty, debug), nprocs=n_cores)


if __name__ == "__main__":
    fire.Fire(main)
