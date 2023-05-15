# evaluate the base gpt2
# n_layer=24, n_head=16, n_embd=1024
# 350M parameters
batch_size = 16
wandb_log = True
wandb_project = 'inshrinkarator_nanogpt_eval'
init_from='resume'
# ckpt_path='/coc/scratch/inshrinkarator/checkpoints/inshrinkarator-nanogpt-inshrinkarator-final-2/gpt2-medium_every_500ba_30k/ckpt_step_30000.pt'
# wandb_run_name='eval_inshrinkarator-final-2/gpt2-medium_every_500ba_30k/ckpt_step_30000'
