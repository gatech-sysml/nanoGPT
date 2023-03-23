# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# inshrinkarator paths, wandb etc
always_save_checkpoint = True
wandb_log = True
wandb_project = 'inshrinkarator_nanogpt'
wandb_run_name='gpt2-medium'
out_dir= '/serenity/scratch/inshrinkarator/checkpoints/nanogpt_3-21/gpt2-medium_every_200ba'
assert out_dir is not '', "out_dir not set"

# these make the total batch size be ~0.6M
# 18 batch size * 1024 block size * 4 gradaccum * 8 GPUs = 589,824
batch_size = 18
block_size = 1024
gradient_accumulation_steps = 4

# this makes total number of tokens be ~4.7B
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100

# eval stuff
eval_interval = 200
eval_iters = 100    # number of batches to for estimating eval and train losses
log_interval = 10

# weight decay
weight_decay = 1e-1

# architecture for gpt-2-medium with 335M parameters
n_layer = 24
n_head = 16
n_embd = 1024

# learning rates
learning_rate = 4e-4 #
min_lr = 4e-5