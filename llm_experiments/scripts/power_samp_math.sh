MODE=${1:-uniform} 

cd /graft2/code/emmanuel/reasoning-with-sampling/llm_experiments

mkdir -p logs/math

CUDA_VISIBLE_DEVICES=0,1 nohup python power_samp_math.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=uniform \
    > logs/math/uniform_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,0 nohup python power_samp_math.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=uniform \
    > logs/math/uniform_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python power_samp_math.py \
    --batch_idx=2 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=uniform \
    > logs/math/uniform_3.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python power_samp_math.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=priority \
    > logs/math/priority_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python power_samp_math.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=priority \
    > logs/math/priority_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python power_samp_math.py \
    --batch_idx=2 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=priority \
    > logs/math/priority_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python power_samp_math.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=restart \
    > logs/math/restart_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python power_samp_math.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=restart \
    > logs/math/restart_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python power_samp_math.py \
    --batch_idx=2 --mcmc_steps=10 --batch_size=167 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=restart \
    > logs/math/restart_3.log 2>&1 &