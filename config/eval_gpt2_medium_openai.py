# evaluate the base gpt2
# n_layer=24, n_head=16, n_embd=1024
# 350M parameters
batch_size = 16
eval_iters = 250 # use more iterations to get good estimate
eval_only = True

dataset = 'lambada'

wandb_log = True
wandb_project = 'inshrinkarator_nanogpt'
init_from = 'gpt2-medium'
wandb_run_name=f'eval_{dataset}_gpt2-medium-openai'