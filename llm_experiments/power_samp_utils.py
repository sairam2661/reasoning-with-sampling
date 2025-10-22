import os

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

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha} 


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


def _resample_idx_distribution(
    propose_style: str,
    current_ids: list,
    log_probs_norm: list,
    c: int,
) -> np.ndarray:
    seq_len = len(current_ids)
    actual_weights_len = len(log_probs_norm)
    
    if propose_style == "restart":
        resample_distr = np.zeros(seq_len)
        resample_distr[c] = 1.0
        
    elif propose_style == "uniform":
        resample_distr = np.zeros(seq_len)
        resample_distr[c:] = 1.0 / (seq_len - c)
        
    elif propose_style == "priority":
        resample_distr = np.zeros(seq_len)
        
        weights = np.exp(-np.array(log_probs_norm))
        weights = weights - 1
        weights = np.maximum(weights, 0)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(actual_weights_len) / actual_weights_len
        
        resample_distr[c:c+actual_weights_len] = weights
        
    else:
        raise ValueError(f"Unknown proposal style: {propose_style}")
    
    assert np.isclose(resample_distr.sum(), 1.0), f"Distribution doesn't sum to 1: {resample_distr.sum()}"
    return resample_distr


def compute_proposal_logprob(current_ids: list, next_ids: list, log_probs_norm_current: list, log_probs_norm_next: list, propose_style: str, c: int):
    resample_idx_distr = _resample_idx_distribution(
        propose_style, current_ids, log_probs_norm_current, c
    )
    
    lcp_idx = c  # Start from end of the end
    for i in range(c, min(len(current_ids), len(next_ids))):
        if current_ids[i] == next_ids[i]:
            lcp_idx = i + 1
        else:
            break
    
    max_resample_idx = min(lcp_idx + 1, len(current_ids))
    
    proposal_logprob = -np.inf
    for idx in range(c, max_resample_idx):
        idx_prob = resample_idx_distr[idx]
        if idx_prob == 0:
            continue
        
        idx_logprob = np.log(idx_prob)
        suffix_logprob = sum(log_probs_norm_next[idx - c:])        
        proposal_logprob = np.logaddexp(proposal_logprob, idx_logprob + suffix_logprob)
    
    return proposal_logprob


# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p+logit_q

# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

# low-temperature sampling proposal distribution
def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]


    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


# alpha = infty power sampling; temp is for proposal distribution
def max_swap(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

# power sampling with autoregressive mcmc
def mcmc_power_samp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16, proposal_type="uniform"):
    c = len(context)
    print(f'alpha: {1/temp}')
    print(f'proposal_type: {proposal_type}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0

    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            
            # Select resampling index based on proposal type
            if proposal_type == "uniform":
                # uniform random
                idx = random.randint(c, t-1)
                
                print(f'Resampling from index {idx} (proposal: {proposal_type})')
                
                prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
                s = len(prop)
                assert(len(log_prob_prop) == s - idx)
                assert(len(target_log_prob_prop) == s - idx)
                log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
                target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
                
                log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)
                
            elif proposal_type == "restart":
                idx = c
                
                print(f'Resampling from index {idx} (proposal: {proposal_type})')
                
                prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
                s = len(prop)
                assert(len(log_prob_prop) == s - idx)
                assert(len(target_log_prob_prop) == s - idx)
                log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
                target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
                
                # For restart, q(curr|prop) = q(prop|curr) = 1
                log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)
                
            elif proposal_type == "priority":
                resample_idx_distr = _resample_idx_distribution(
                    proposal_type, gen, log_probs_norm, c
                )
                idx = np.random.choice(len(gen), p=resample_idx_distr)
                
                print(f'Resampling from index {idx} (proposal: {proposal_type})')
                
                prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
                s = len(prop)
                assert(len(log_prob_prop) == s - idx)
                assert(len(target_log_prob_prop) == s - idx)
                log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
                target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
                
                # q(prop | curr)
                prop_logprob_curr_to_next = compute_proposal_logprob(
                    gen, prop, log_probs_norm, log_prob_prop, proposal_type, c
                )
                
                # q(curr | prop)
                prop_logprob_next_to_cur = compute_proposal_logprob(
                    prop, gen, log_prob_prop, log_probs_norm, proposal_type, c
                )
                
                # Acceptance ratio
                log_r = sum(target_log_prob_prop) + prop_logprob_next_to_cur - \
                        sum(target_log_prob_cur) - prop_logprob_curr_to_next

            # Accept or reject
            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
