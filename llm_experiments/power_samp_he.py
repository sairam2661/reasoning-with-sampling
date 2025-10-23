import os

from huggingface_hub import constants
import re

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_grpo", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", action = "store", default = 0.5, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "HUMANEVAL", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--type", action = "store", type = str, default = "chat", choices = ["chat"])
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--batch_size", action="store", type=int, default=100, help="Number of problems per batch")
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--proposal_type", action="store", type=str, default="uniform", choices=["uniform", "priority", "restart"], help="Proposal distribution for MCMC: uniform (prefix), priority (perplexity), or restart")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline methods")

    args = parser.parse_args()

    random.seed(args.seed)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model, "he")
    os.makedirs(save_str, exist_ok=True)


    print(model)
    print(device)
    print(mcmc_steps)
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = 'microsoft/Phi-3.5-mini-instruct'
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"

    if dataset_name == "HUMANEVAL":
        json_file = 'data/HumanEval.jsonl'
        with open(json_file, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]


    print("dataset done")

    config = transformers.AutoConfig.from_pretrained(model_str, trust_remote_code=False, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=False, local_files_only=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, config=config, torch_dtype="auto", device_map="auto", trust_remote_code=False, local_files_only=True).to(device)

    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("loaded models")
    
    start = args.batch_idx * args.batch_size
    end = min(start + args.batch_size, len(dataset))
    
    print(f"Processing problems {start} to {end-1} (batch_idx={args.batch_idx}, batch_size={args.batch_size})")

    output_file = os.path.join(save_str,
        f"base_power_samp_results_{mcmc_steps}_{args.proposal_type}_{temp}_{args.batch_idx}_{args.seed}.csv")
    
    if os.path.exists(output_file):
        print(f"Found existing results at {output_file}, loading...")
        df_existing = pd.read_csv(output_file)
        results = df_existing.to_dict('records')
        completed_problems = len(results)
        print(f"Resuming from problem {completed_problems} within this batch")
    else:
        results = []
        completed_problems = 0

    for problem_idx in tqdm(range(start, end), desc=f"Batch {args.batch_idx}"):
         # Skip already completed problems
        local_idx = problem_idx - start 
        if local_idx < completed_problems:
            continue
        
        data = dataset[problem_idx]
        prompt = data["prompt"]
        task_id = data["task_id"]

        if model == "phi" or model == "phi_grpo":
            signature = re.search(
                rf"def\s+({data['entry_point']}.*?):\s*\n", data["prompt"]
            ).group(1)
            description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", data["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
            input_text = (
                f"Write a Python function `{signature}` to solve the following problem:\n"
                f"{description}\n"
                f"{data['prompt']}"
            )

        else:
            input_text = prompt

        print(input_text)


        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        if not args.skip_baselines:  
            naive_temp_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                    return_dict_in_generate=True, output_scores=True, temperature = temp)
            
            print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
            print("naive done")
            
            
            std_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                    return_dict_in_generate=True, output_scores=True, do_sample = True)
            
            print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
            print("std done")
        else:
            naive_temp_output = None
            std_output = None

        mcmc_temp_output, _, _, acceptance_ratio = mcmc_power_samp(autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072, proposal_type=args.proposal_type)

        if not args.skip_baselines:
            print(len(std_output))
            print(len(naive_temp_output))
        print(len(mcmc_temp_output))
        print(tokenizer.decode(torch.tensor([mcmc_temp_output], dtype=torch.long, device=device).squeeze().to("cpu"), skip_special_tokens=True))
        print("mcmc done")

        if not args.skip_baselines:
            naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
            std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")

            naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
            std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)

            naive_answer = parse_answer(naive_completion)
            std_answer = parse_answer(std_completion)
        else:
            naive_completion = ""
            std_completion = ""
            naive_answer = ""
            std_answer = ""
            
        mcmc_temp_ids = torch.tensor([mcmc_temp_output], dtype=torch.long, device=device).squeeze().to("cpu")
        mcmc_completion = tokenizer.decode(mcmc_temp_ids, skip_special_tokens=True)
        mcmc_answer = parse_answer(mcmc_completion)

        print(f'Acceptance: {acceptance_ratio}')


        results.append({
            "problem_idx": problem_idx,
            "question": prompt,
            "id": task_id,
            "naive_completion": naive_completion,
            "std_completion": std_completion,
            "mcmc_completion": mcmc_completion,
            "acceptance_ratio": acceptance_ratio,
            "proposal_type": args.proposal_type,
        })

    
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file} ({len(results)} problems completed in this batch)")

    print(f"\n{'='*80}")
    print(f"Batch {args.batch_idx} completed! Processed problems {start}-{end-1}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")











        













