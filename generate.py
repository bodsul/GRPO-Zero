import dataclasses
import gc
import math
import yaml
from pathlib import Path

from collections import defaultdict
from typing import Callable, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_types import MiniBatch, QMiniBatch
from qwen2_model import Transformer, Qwen2Config
from tokenizer import Tokenizer
from gsm8k_task import Gsm8k_Task_Dataset, reward_function_dispatcher
from generator import TokensGenerator

class GreedyGenerator(TokensGenerator):
    def __init__(self, model: Transformer,
        batch: Union[MiniBatch, QMiniBatch],
        tokenizer: Tokenizer,
        max_gen_len: int,
        num_answer_per_question: int,
        device: torch.device,
        dtype: torch.dtype,):
        super(GreedyGenerator, self).__init__(model, batch, tokenizer, max_gen_len, num_answer_per_question, device, dtype)

    @torch.no_grad()
    def generate_tokens(self):
        for cur_pos in range(self.min_prompt_len, self.total_len):
            probs = self.get_probs(cur_pos)
            next_token = torch.argmax(probs, dim=-1)
            next_token = next_token.reshape(-1)
            if not self.update_tokens(next_token, cur_pos):
                break
        self.cleanup()

    def get_processed_generated_tokens(self, **kwargs)-> List:
        self.generate_tokens()
        pad_token_id = self.tokenizer.pad_token_id
        end_token = self.tokenizer.eos_token

        responses = []
        tokens_list = self.tokens.tolist()

        for i in range(self.bsz):
            generated_token_ids = tokens_list[i][len(self.batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = self.tokenizer.detokenize(generated_token_ids)
            responses.append([{'response': generated_text, 'score': None}])
        
        return responses

class BeamSearchGenerator(TokensGenerator):
    def __init__(self, model: Transformer,
        batch: MiniBatch,
        tokenizer: Tokenizer,
        max_gen_len: int,
        num_answer_per_question: int,
        device: torch.device,
        dtype: torch.dtype,):
        super(BeamSearchGenerator, self).__init__(model, batch, tokenizer, max_gen_len, num_answer_per_question, device, dtype)
        self.running_log_p_sum = None

    @torch.no_grad()
    def generate_tokens(self):
        num_hypothesis = self.num_answer_per_question
        self.running_log_p_sum = torch.zeros((self.bsz,), dtype=self.dtype, device=self.device)
        vocab_size = Qwen2Config.vocab_size
        vocab_mask = torch.zeros((len(self.batch.prefix), vocab_size*self.num_answer_per_question), dtype=torch.bool, device=self.device)
        vocab_mask[:, vocab_size:] = True
        
        for cur_pos in range(self.min_prompt_len, self.total_len):
            probs = self.get_probs(cur_pos)
            if len(probs[self.is_finished]) != 0:
                probs[self.is_finished] = 0 
                probs[self.is_finished][:, self.tokenizer.eos_token_id] = 1
            
            log_probs_tot = torch.log(probs)
            log_probs_tot += self.running_log_p_sum.unsqueeze(-1)
            log_probs_tot = log_probs_tot.reshape(-1)
            log_probs_tot = log_probs_tot.reshape(len(self.batch.prefix), num_hypothesis*vocab_size)
            log_probs_tot = torch.where(self.input_text_mask[::num_hypothesis, cur_pos-1].unsqueeze(-1) & vocab_mask, -torch.inf, log_probs_tot)
            top_k = torch.topk(log_probs_tot, k=num_hypothesis, dim=-1) # (b, n)

            top_k_indices = top_k.indices
            top_k_values = top_k.values
            top_k_vocab_indices = top_k_indices % vocab_size
            top_k_row_indices = top_k_indices // vocab_size + (torch.arange(len(self.batch.prefix))*num_hypothesis).unsqueeze(-1)

            top_k_vocab_indices = top_k_vocab_indices.reshape(-1)
            top_k_row_indices = top_k_row_indices.reshape(-1)
            top_k_values = top_k_values.reshape(-1)

            self.running_log_p_sum = top_k_values
            self.running_log_p_sum = torch.where(self.input_text_mask[:, cur_pos], 0, self.running_log_p_sum)
            self.tokens = self.tokens[top_k_row_indices]
            self.model.update_kv_cache_for_beam_search(top_k_row_indices)
            self.is_finished = self.is_finished[top_k_row_indices]
            next_token = top_k_vocab_indices

            if not self.update_tokens(next_token, cur_pos):
                break
        self.cleanup()

    def get_processed_generated_tokens(self, **kwargs)-> List:
        self.generate_tokens()
        pad_token_id = self.tokenizer.pad_token_id
        end_token = self.tokenizer.eos_token
        tokens_list = self.tokens.tolist()
        num_hypothesis = self.num_answer_per_question

        responses = []

        for i in range(len(self.batch.prefix)):
            this_responses = []
            for j in range(num_hypothesis):
                k = i*num_hypothesis + j
                generated_token_ids = tokens_list[k][len(self.batch.prefix_token_ids[i]) :]
                # remove padding tokens
                if pad_token_id in generated_token_ids:
                    generated_token_ids = generated_token_ids[
                        : generated_token_ids.index(pad_token_id)
                    ]
                generated_text = self.tokenizer.detokenize(generated_token_ids)
                this_responses.append({'response': generated_text, 'score': self.running_log_p_sum[k].item()})
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
    model.load_state_dict(torch.load(output_file))
    max_gen_len=config["training"]["max_gen_len"] * 2
    tot_reward = 0
    n_instances = 0
    for j, batch in enumerate(dataloader):
        tokens_generator = BeamSearchGenerator(model, batch, tokenizer, max_gen_len, 1, device, dtype)
        responses = tokens_generator.get_processed_generated_tokens()
        for i in range(len(batch.prefix)):
            responses[i].sort(key=lambda x: x['score'])
            best_response = (responses[i][-1]['response'])

            rewards = reward_function_dispatcher(
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