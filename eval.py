"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
from tqdm import tqdm
import os
import time
import math
import pickle
from contextlib import nullcontext
import wandb

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import sys
from os.path import abspath
repo_path = abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_path)

from inshrinkarator.core.compressor_registry import CompressorRegistry
from inshrinkarator.core.types import System
from inshrinkarator.utils.wandb_logger import WandbLogger
from inshrinkarator.utils.stop_signal_handler import StopSignalHandler

from model import GPTConfig, GPT
os.environ["NCCL_P2P_LEVEL"] = "NVL"

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_data_path = 'data/common/wikitext/wikitext-2-raw-v1/test.bin'
eval_dataset_name = 'wikitext-2'
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
ckpt_path ='tmp/'
out_dir = ''
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0

if master_process:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# wandb setup
if wandb_log and master_process:
    wandb_run_name += f'_{eval_dataset_name}'
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# poor man's data loader
val_data = np.memmap(eval_data_path, dtype=np.uint16, mode='r')
# attempt to derive vocab_size from the dataset
data_dir = os.path.dirname(eval_data_path)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
checkpoint = None
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {ckpt_path}")
    # resume training from a checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)
# compile the model
unoptimized_model = None
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    if unoptimized_model:
        unoptimized_model = DDP(unoptimized_model, device_ids=[ddp_local_rank])

# Perplexity
model.eval()
max_length = block_size
stride = block_size // 2
seq_len = len(val_data)
if master_process:
    print("seq_len", seq_len)

nlls = []
prev_end_loc = 0
ix = range(0, seq_len, stride)
with tqdm(ix) as pbar:
    # pbar.set_description(f"perplexity: {perplexity:.2f}")
    for i, begin_loc in enumerate(ix):
        end_loc = begin_loc + max_length
        if end_loc > seq_len:
            break
        # trg_len = end_loc - prev_end_loc

        # input and target ids
        x = torch.from_numpy(val_data[begin_loc:end_loc].astype(np.int64))
        y = torch.from_numpy(val_data[begin_loc + 1:end_loc + 1].astype(np.int64))
        y[:-stride] = -1 # don't include context in loss calculation
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        # move to gpu
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        # forward the model
        with torch.no_grad():
            logits, loss = model(x, y)

        nlls.append(loss)
        if master_process and i % 10 == 0:
            perplexity = torch.exp(torch.stack(nlls).mean())
            pbar.set_description(
                # f"begin_loc {begin_loc} "
                f"perplexity: {perplexity:.2f}"
            )
            pbar.update(10)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

ppl = torch.exp(torch.stack(nlls).mean())
if master_process:
    print("Final Perplexity", ppl.item())
    wandb.log({'perplexity': ppl.item()})

if ddp:
    destroy_process_group()
