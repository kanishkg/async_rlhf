export WANDB_PROJECT=pi-sos
export WANDB_ENTITY=cocolab
accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision bf16 --num_processes 2 rloo.py --config configs/rloo_llama3-8b_countdown.yml
