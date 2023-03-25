# inshrinkarator paths, wandb etc
always_save_checkpoint = False
wandb_log = True
wandb_project = 'inshrinkarator_nanogpt'
wandb_run_name='gpt2-medium-resume22k'
ckpt_path = '/coc/scratch/inshrinkarator/checkpoints/nanogpt_3-21/gpt2-medium_every_200ba_resume10k_2/ckpt_step_22000.pt'
out_dir= '/serenity/scratch/inshrinkarator/checkpoints/nanogpt_3-21/gpt2-medium_every_200ba_resume22k'
init_from='resume'
assert out_dir != '', "out_dir not set"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 4

max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 0
# eval stuff
eval_interval = 200
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-1

# architecture for gpt 2 mmedium
n_layer = 24
n_head = 16
n_embd = 1024

# learning rates
learning_rate = 2e-4 #
min_lr = 4e-5
