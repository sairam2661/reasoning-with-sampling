MODE=${1:-uniform} 

cd /data/saiva/reasoning-with-sampling/llm_experiments

mkdir -p logs/he

export PYTHONPATH="$PYTHONPATH:/data/saiva/reasoning-with-sampling/llm_experiments"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python power_samp_he.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=250 --temp=0.25 --seed=42 \
    --model=qwen_math --proposal_type=$MODE \
    > logs/he/${MODE}_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python power_samp_he.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=250 --temp=0.25 --seed=42 \
    --model=qwen_math --proposal_type=$MODE \
    > logs/he/${MODE}_2.log 2>&1 &