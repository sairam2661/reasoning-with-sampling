cd /data/saiva/reasoning-with-sampling/llm_experiments

mkdir -p logs

# Launch all three
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python power_samp_math.py --batch_idx=0 --mcmc_steps=10 --temp=0.25 --seed=42 --model=qwen_math --proposal_type=uniform > logs/uniform.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python power_samp_math.py --batch_idx=0 --mcmc_steps=10 --temp=0.25 --seed=42 --model=qwen_math --proposal_type=priority > logs/priority.log 2>&1 &