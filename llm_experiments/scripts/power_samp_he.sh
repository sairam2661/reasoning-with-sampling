MODE=${1:-uniform} 

cd /graft2/code/emmanuel/reasoning-with-sampling/llm_experiments

mkdir -p logs/he

CUDA_VISIBLE_DEVICES=0 nohup python power_samp_he.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=82 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=uniform \
    > logs/he/uniform_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python power_samp_he.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=82 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=uniform \
    > logs/he/uniform_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python power_samp_he.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=82 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=priority \
    > logs/he/priority_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python power_samp_he.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=82 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=priority \
    > logs/he/priority_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python power_samp_he.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=82 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=restart \
    > logs/he/restart_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python power_samp_he.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=82 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=restart \
    > logs/he/restart_2.log 2>&1 &