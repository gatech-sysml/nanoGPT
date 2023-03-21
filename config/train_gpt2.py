# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'inshrinkarator_nanogpt'
wandb_run_name='gpt2-medium'
out_dir= '/serenity/scratch/inshrinkarator/checkpoints/nanogpt_3-21/gpt2-medium_every_200ba'
assert out_dir is not '', "out_dir not set"
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 18
block_size = 1024
gradient_accumulation_steps = 4

# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100
# eval stuff
eval_interval = 200
eval_iters = 10
log_interval = 10

# weight decay
weight_decay = 1e-1

# architecture for gpt 2 mmedium
n_layer = 24
n_head = 16
n_embd = 1024

# learning rates
learning_rate = 4e-4 #
min_lr = 4e-5