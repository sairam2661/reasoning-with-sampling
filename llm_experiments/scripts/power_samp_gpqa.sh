MODE=${1:-uniform} 

cd /data/saiva/reasoning-with-sampling/llm_experiments

mkdir -p logs/gpqa

export PYTHONPATH="$PYTHONPATH:/data/saiva/reasoning-with-sampling/llm_experiments"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python power_samp_gpqa.py \
    --batch_idx=0 --mcmc_steps=10 --batch_size=99 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=$MODE \
    > logs/gpqa/${MODE}_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python power_samp_gpqa.py \
    --batch_idx=1 --mcmc_steps=10 --batch_size=99 --temp=0.25 --seed=42 \
    --model=phi --proposal_type=$MODE \
    > logs/gpqa/${MODE}_2.log 2>&1 &