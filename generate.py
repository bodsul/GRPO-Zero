import dataclasses
import gc
import math
import yaml
from pathlib import Path

from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_types import Episode, QMiniBatch
from qwen2_model import Transformer, Qwen2Config
from tokenizer import Tokenizer
from gsm8k_task import Gsm8k_Task_Dataset, gsm8k_reward_function_dispatcher

@torch.no_grad()
def generate(model: Transformer,
    batch: QMiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[str]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix)
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        tokens[k, : len(t)] = torch.tensor(
            t, dtype=torch.long, device=device
        )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating response: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if a generation is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    responses = []

    for i in range(bsz):
        # first_generated_token = tokens_list[i][len(batch.prefix_token_ids[i])]
        generated_token_ids = tokens_list[i][len(batch.prefix_token_ids[i]) :]
        # remove padding tokens
        if pad_token_id in generated_token_ids:
            generated_token_ids = generated_token_ids[
                : generated_token_ids.index(pad_token_id)
            ]
        generated_text = tokenizer.detokenize(generated_token_ids)
        # if generated_text == '':
        #     print('Fail', batch.prefix[i])
        # else:
        #     print('Pass', batch.prefix[i])
        responses.append([{'response': generated_text, 'score': 0}])
    
    # exit()
    return responses

@torch.no_grad()
def generate_beam_search(model: Transformer,
    batch: QMiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    device: torch.device,
    dtype: torch.dtype,
    num_hypothesis: int,
) -> List[str]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix)*num_hypothesis
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k*num_hypothesis
        for i in range(num_hypothesis):
            tokens[offset+i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)
    
    running_log_p_sum = torch.zeros((bsz,), dtype=dtype, device=device)
    vocab_size = Qwen2Config.vocab_size
    vocab_mask = torch.zeros((len(batch.prefix), vocab_size*num_hypothesis), dtype=torch.bool, device=device)
    vocab_mask[:, vocab_size:] = True

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating response: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1) # (b*n, v)
        if len(probs[is_finished]) != 0:
            probs[is_finished] = 0 
            probs[is_finished][:, end_token_id] = 1
        log_probs_tot = torch.log(probs)
        log_probs_tot += running_log_p_sum.unsqueeze(-1)
        log_probs_tot = log_probs_tot.reshape(-1)
        log_probs_tot = log_probs_tot.reshape(len(batch.prefix), num_hypothesis*vocab_size)
        log_probs_tot = torch.where(input_text_mask[::num_hypothesis, cur_pos-1].unsqueeze(-1) & vocab_mask, -torch.inf, log_probs_tot)
        top_k = torch.topk(log_probs_tot.unsqueeze(-1), k=num_hypothesis, dim=-1) # (b, n)

        top_k_indices = top_k.indices
        top_k_values = top_k.values
        # print(top_k_values)
        # print(vocab_size)
        # print(top_k_indices)
        top_k_vocab_indices = top_k_indices % vocab_size
        top_k_row_indices = top_k_indices // vocab_size + (torch.arange(len(batch.prefix))*num_hypothesis).unsqueeze(-1)

        top_k_vocab_indices = top_k_vocab_indices.reshape(-1)
        top_k_row_indices = top_k_row_indices.reshape(-1)
        # print(top_k_row_indices)
        top_k_values = top_k_values.reshape(-1)

        running_log_p_sum = top_k_values
        tokens = tokens[top_k_row_indices]
        model.update_kv_cache_for_beam_search(top_k_row_indices)
        is_finished = is_finished[top_k_row_indices]
        next_token = top_k_vocab_indices
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        running_log_p_sum = torch.where(input_text_mask[:, cur_pos], 0, running_log_p_sum)
        # if a generation is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    responses = []

    for i in range(len(batch.prefix)):
        this_responses = []
        for j in range(num_hypothesis):
            k = i*num_hypothesis + j
            generated_token_ids = tokens_list[k][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            this_responses.append({'response': generated_text, 'score': running_log_p_sum[k].item()})
        responses.append(this_responses)
    
    return responses

def run(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    ckpt_dir = Path(config["training"]["ckpt_dir"])

    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    BATCH_SIZE = config["training"]["batch_size"]

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    dataset = Gsm8k_Task_Dataset(
    data_path=config["data"]["path"],
    tokenizer=tokenizer,
    split="test",
    test_size=config["data"]["test_size"],
)
    dataloader = DataLoader(
        dataset,
        collate_fn=Gsm8k_Task_Dataset.collate_fn,
        batch_size=BATCH_SIZE//4,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    step = 200
    output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
    # model.load_state_dict(torch.load(output_file))
    max_gen_len=config["training"]["max_gen_len"] * 2
    tot_reward = 0
    n_instances = 0
    for j, batch in enumerate(dataloader):
        responses = generate(model, batch, tokenizer, max_gen_len, device, dtype)
        # responses = generate_beam_search(model, batch, tokenizer, max_gen_len, device, dtype, 4)
        for i in range(len(batch.prefix)):
            responses[i].sort(key=lambda x: x['score'])
            best_response = (responses[i][-1]['response'])

            rewards = gsm8k_reward_function_dispatcher(
                    response=best_response,
                    batch=batch,
                    end_token=tokenizer.eos_token,
                    i=i
                )
            tot_reward+=rewards["reward_info"]["answer_reward"]
            n_instances+=1
        print(f'answer_reward: {rewards["reward_info"]["answer_reward"]}')
        print(f'curr_avg_reward: {tot_reward/n_instances}')
    print(f'avg_reward: {tot_reward/n_instances}')


if __name__=='__main__':
    run('gsm8k_config.yaml')