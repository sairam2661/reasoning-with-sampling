# Reasoning with Sampling

### [Paper]() | [Project Page](https://aakaran.github.io/training_free_reasoning/)


This repo contains the official PyTorch implementation of Reasoning with Sampling.
> [**Reasoning with Sampling: Your Base Model is Smarter Than You Think**]()<br>
> [Aayush Karan](https://aakaran.github.io/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

Run the following script to setup environment.

```bash
git clone https://github.com/aakaran/reasoning-with-sampling.git
cd reasoning-with-sampling
conda create -n psamp python=3.10
conda activate psamp
pip install -r requirements.txt
```


## Sampling
The llm_experiments folder contains .py scripts to run power sampling for MATH500 (power_samp_math.py), whose .json is included in llm_experiments/data, as well as HumanEval (power_samp_he.py) and GPQA Diamond (power_samp_gpqa.py), whose corresponding eval sets can be downloaded from their official repos. 

To run power sampling on MATH500 with 8 seeds and splitting the dataset across 5 shards:
```bash
sbatch llm_experiments/scripts/power_samp_math.sh
```
The output is several .csv files (based on the shard and seed number) that store the response outputs, correct answers, original prompts, etc.

## Evaluation
**Single-shot Reasoning**

**Pass@k Performance**

