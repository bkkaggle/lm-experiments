import fire

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from finetune import finetune


def main(**kwargs):
    xmp.spawn(finetune, args=(kwargs))


if __name__ == "__main__":
    fire.Fire(main)
