# evaluate the base gpt2
# n_layer=24, n_head=16, n_embd=1024
# 350M parameters
batch_size = 16
wandb_log = True
wandb_project = 'inshrinkarator_nanogpt_eval'
compile = False

# Pass these as arguments
# init_from = 'gpt2-medium'
# wandb_run_name='gpt2-medium-openai'
# For example:
# torchrun --standalone --nproc_per_node=2 eval.py config/eval_gpt2_openai.py --init_from=gpt2-medium --wandb_run_name='gpt2-medium-openai'