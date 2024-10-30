from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("if001/oscar_2023_filtered")['train']
dataset = dataset.select(range(len(dataset) // 5))
text_column = "text"
output_file = "oscar_train_tok.txt"

with open(output_file, "w") as f:
    for sample in tqdm(dataset[text_column]):
        f.write(sample + "<|endoftext|>")