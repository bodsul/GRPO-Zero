import re
from typing import Any, Dict, List, Optional

from task_data import Task_Dataset

from data_types import QMiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "{question} "
    "The final answer should be prefixed with #### and should appear after the reasoning." 
)
RESPONSE_PROMPT = "Let me solve this step by step."

class Gsm8k_Task_Dataset(Task_Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
       super(Gsm8k_Task_Dataset, self).__init__(tokenizer, data_path, 'main', split, test_size)
    

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["question"]))
        return item

    def encode_prefix(self, question: str):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(question=question)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> QMiniBatch:
        """Collate examples into a batch."""
        question = [item["question"] for item in batch]
        target = [item["answer"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return QMiniBatch(
            question=question,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, end_token=None, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    ground_truth = extract_solution(solution_str=ground_truth, method='strict')
    assert ground_truth is not None

    # Strip end token if present
    if end_token and solution_str.endswith(end_token):
        solution_str = solution_str[: -len(end_token)]

    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        r_score, r_format_score=0, 0
    else:
        if answer == ground_truth:
            r_score, r_format_score=score, format_score
        else:
            r_score, r_format_score=0, format_score

    return {
        "reward": r_score+r_format_score,
        "reward_info": {
            "format_reward": r_format_score,
            "answer_reward": r_score,
        },
    }

def reward_function_dispatcher(response, batch, end_token, i):
    return compute_score(
                solution_str=response,
                ground_truth=batch.target[i],
                end_token=end_token,
            )