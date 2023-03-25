# evaluate the base gpt2
# n_layer=24, n_head=16, n_embd=1024
# 350M parameters
batch_size = 16
eval_iters = 1000 # use more iterations to get good estimate
eval_only = True

dataset = 'lambada'

wandb_log = True
wandb_project = 'inshrinkarator_nanogpt'
init_from='resume'
ckpt_path='/coc/scratch/inshrinkarator/checkpoints/nanogpt_3-21/gpt2-medium_every_200ba/ckpt_step_10000.pt'
wandb_run_name=f'eval_gpt2-medium-mine'
