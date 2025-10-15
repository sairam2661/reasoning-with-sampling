# Reasoning without Training

### [Paper]() | [Project Page](https://aakaran.github.io/training_free_reasoning/)


> [**Reasoning Without Training: Your Base Model is Smarter Than You Think**]()<br>
> [Aayush Karan](https://aakaran.github.io/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

Run the following script to setup environment.

```bash
git clone https://github.com/aakaran/training-free-reasoning.git
cd training-free-reasoning
conda create -n psamp python=3.10
conda activate psamp
pip install -r requirements.txt
```


## Sampling

**.** To run power sampling on MATH500 :
```bash
python llm_experiments/power_samp_math.py --model --
```

## Evaluation

