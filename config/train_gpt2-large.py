# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# inshrinkarator paths, wandb etc
always_save_checkpoint = True
wandb_log = True
wandb_project = 'inshrinkarator_nanogpt'
wandb_run_name='gpt2-large_30k'
out_dir= '/serenity/scratch/inshrinkarator/checkpoints/nanogpt_3-21/gpt2-large_every_500ba_30k'
assert out_dir != '', "out_dir not set"

# these make the total batch size be ~0.6M
# 8 batch size * 1024 block size * 8 gradaccum * 8 GPUs = ~524M
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 11

# this makes total number of tokens be ~15B
max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 300

# eval stuff
eval_interval = 500
eval_iters = 100    # number of batches to for estimating eval and train losses
log_interval = 10

# weight decay
weight_decay = 1e-1

# architecture for gpt-2-medium with 335M parameters
n_layer = 36
n_head = 20
n_embd = 1280

# learning rates
learning_rate = 4e-4 #
min_lr = 4e-5
