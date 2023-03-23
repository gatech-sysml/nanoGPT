# saves any huggingface dataset to a binary file for training. following was helpful:
# Recommended datasets to try out:
# - openwebtext
# - lambada
# - wikitext/wikitext2-v1
# - wikitext/wikitext103-v1
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import argparse
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help="dataset to use. for subsets like wikitext use with slash like 'wikitext/wikitext-2-raw-v1'")
parser.add_argument('--out_root', type=str, help='output directory root', default='/serenity/data/datasets/')
args = parser.parse_args()

# make output directory
out_dir = os.path.join(args.out_root, args.dataset)
os.makedirs(out_dir, exist_ok=True)
print(f"out_dir: {out_dir}")

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
if '/' in args.dataset:
    d = args.dataset.split('/')
    assert len(d) == 2
    dataset = load_dataset(*d)
else:
    dataset = load_dataset(args.dataset)

# Split dataset into train and val if it doesn't already have a val split
if 'validation' in dataset:
    dataset['val'] = dataset.pop('validation')
    split_dataset = dataset
if 'val' not in dataset:
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

print(f"split_dataset: {split_dataset}")
# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(out_dir, f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
