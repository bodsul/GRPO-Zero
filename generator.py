import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List, Union, Any

import numpy as np
import torch

from qwen2_model import Transformer
from tokenizer import Tokenizer
from data_types import MiniBatch, QMiniBatch

class TokensGenerator:
    'Common functionalities for tokens generation'
    def __init__(self, model: Transformer,
        batch: Union[MiniBatch, QMiniBatch],
        tokenizer: Tokenizer,
        max_gen_len: int,
        num_answer_per_question: int,
        device: torch.device,
        dtype: torch.dtype,):
        self.batch = batch
        self.tokenizer = tokenizer
        self.bsz = len(batch.prefix) * num_answer_per_question
        self.min_prompt_len = min(len(t) for t in self.batch.prefix_token_ids)
        self.max_prompt_len = max(len(t) for t in self.batch.prefix_token_ids)
        self.max_gen_len = max_gen_len
        self.total_len = self.max_gen_len + self.max_prompt_len
        self.num_answer_per_question = num_answer_per_question
        self.model = model

        self.model.init_kv_cache(
            max_batch_size=self.bsz,
            max_seq_len=self.total_len,
            device=device,
            dtype=dtype,
        )
        self.tokens = torch.full((self.bsz, self.total_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        
        for k, t in enumerate(self.batch.prefix_token_ids):
            offset = k * num_answer_per_question
            for i in range(num_answer_per_question):
                self.tokens[offset + i, : len(t)] = torch.tensor(
                    t, dtype=torch.long, device=device
                )

        self.prev_pos = 0
        self.input_text_mask = self.tokens != self.tokenizer.pad_token_id
        assert self.min_prompt_len < self.total_len
        self.is_finished = torch.zeros((self.bsz,), dtype=torch.bool, device=device)
        self.dtype = dtype
        self.device = device

    def update_tokens(self, next_token, cur_pos):
        pad_token_id = self.tokenizer.pad_token_id
        end_token_id = self.tokenizer.eos_token_id
        next_token = torch.where(
            self.input_text_mask[:, cur_pos], self.tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(self.is_finished, pad_token_id, next_token)
        self.tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~self.input_text_mask[:, cur_pos]
            self.is_finished = self.is_finished | (is_end_token & is_generated_token)
        self.prev_pos = cur_pos
        if self.is_finished.all():
            return False
        else:
            return True

    def cleanup(self):
        self.model.del_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def get_probs(self, cur_pos):
        print(
            f"\r* Generating trajectories: {cur_pos-self.min_prompt_len:>4d}/{self.total_len-self.min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits = self.model.inference(self.tokens[:, self.prev_pos:cur_pos], self.prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        return probs

    @torch.no_grad()
    def generate_tokens(self):
        raise NotImplementedError('Subclass TokensGenerator and implement generate_tokens')

    def get_processed_generated_tokens(self, **kwargs)-> Any:
        raise NotImplementedError('Subclass TokensGenerator and implement get_processed_generated_tokens')