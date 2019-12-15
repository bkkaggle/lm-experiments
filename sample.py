import fire
import numpy as np
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# From: https://github.com/huggingface/transformers/blob/master/examples/run_generation.py#L79


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (logits.dim(
    ) == 1)  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Parts from: https://github.com/huggingface/transformers/blob/master/examples/run_generation.py


def sample(prompt, model, tokenizer, length, temperature, top_k, top_p, repetition_penalty):
    next_token = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = next_token

    past = None
    with torch.no_grad():
        for _ in tqdm(range(length)):
            logits, past = model(next_token, past=past)

            logits = logits.view(-1)
            logits /= temperature if temperature > 0 else 1

            # Repetition penalty
            for i in set(generated.view(-1).tolist()):
                logits[i] /= repetition_penalty

            # Top-k or top-p
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            # Greedy sampling
            if temperature == 0:
                next_token = torch.argmax(logits).unsqueeze(0)
            # Top-k or top-p
            else:
                next_token = torch.multinomial(torch.softmax(
                    logits.float(), dim=-1), num_samples=1)

            generated = torch.cat([generated, next_token.view(1, 1)], dim=1)

        out = tokenizer.decode(generated.view(-1).tolist())

        return out


def main(checkpoint="gpt2", prompt=None, length=100, temperature=0, top_k=0, top_p=0, repetition_penalty=1.2, debug=False):
    if debug:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)

    while True:
        try:
            if prompt == None:
                prompt = input("prompt > ")
            out = sample(prompt, model, tokenizer, length,
                         temperature, top_k, top_p, repetition_penalty)
            print(f"Generated: {out}")

            if prompt != None:
                break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    fire.Fire(main)
