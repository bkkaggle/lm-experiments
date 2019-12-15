import os

import wandb


class Config(object):
    def __init__(self, **kwargs):
        self.dataset_path = kwargs.get('dataset_path', './data.pkl')
        self.save_dir = kwargs.get('save_dir', wandb.run.dir)

        self.model = kwargs.get('model', 'gpt2')
        self.checkpoint = kwargs.get('checkpoint', 'gpt2')

        self.lr = kwargs.get('lr', 5e-5)
        self.batch_size = kwargs.get('batch_size', 4)
        self.gradient_accumulation_steps = kwargs.get(
            'gradient_accumulation_steps', 1)
        self.epochs = kwargs.get('epochs', 10)

        self.accelerator = kwargs.get('accelerator', 'GPU')

        self.logging_steps = kwargs.get('logging_steps', 10)
        self.histogram_steps = kwargs.get('histogram_steps', 100)
        self.save_steps = kwargs.get('save_steps', 100)

        self.n_samples = kwargs.get('n_samples', 1)
        self.sample_len = kwargs.get('sample_len', 256)
        self.temperature = kwargs.get('temperature', 1)
        self.top_k = kwargs.get('top_k', 0)
        self.top_p = kwargs.get('top_p', 0.9)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.2)

        self.debug = kwargs.get('debug', False)

        # Preprocessing config vars
        self.dataset_name = kwargs.get('dataset_name', None)
        self.seq_len = kwargs.get('seq_len', 256)
        self.control_code = kwargs.get('control_code', None)

        wandb.config.dataset_path = self.dataset_path
        wandb.config.save_dir = self.save_dir
        wandb.config.model = self.model
        wandb.config.checkpoint = self.checkpoint
        wandb.config.lr = self.lr
        wandb.config.batch_size = self.batch_size
        wandb.config.gradient_accumulation_steps = self.gradient_accumulation_steps
        wandb.config.epochs = self.epochs
        wandb.config.accelerator = self.accelerator
        wandb.config.logging_steps = self.logging_steps
        wandb.config.histogram_steps = self.histogram_steps
        wandb.config.save_steps = self.save_steps
        wandb.config.n_samples = self.n_samples
        wandb.config.sample_len = self.sample_len
        wandb.config.sample_len = self.sample_len
        wandb.config.temperature = self.temperature
        wandb.config.top_k = self.top_k
        wandb.config.top_p = self.top_p
        wandb.config.repetition_penalty = self.repetition_penalty
        wandb.config.debug = self.debug

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
