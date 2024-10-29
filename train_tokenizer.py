from cs336_basics import BPETok, train_bpe
import cProfile
import json


def train():
    # tokenizer = BPETok.from_train("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens=["<|endoftext|>"])
    tokenizer = BPETok.from_train("data/owt_train.txt", 32000, special_tokens=["<|endoftext|>"])
    # tokenizer = BPETok.from_local("tinystories_tok.json")
    tokenizer.save("owt_tok.json")

if __name__ == '__main__':
    train()