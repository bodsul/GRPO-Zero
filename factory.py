from typing import Callable

from gsm8k_task import Gsm8k_Task_Dataset, reward_function_dispatcher as gsm8k_reward_function_dispatcher
from countdown_task import CountdownTasksDataset, reward_function_dispatcher as countdown_reward_function_dispatcher
from task_data import Task_Dataset


def create_dataset(dataset: str, **kwargs) -> Task_Dataset:
    if dataset == 'countdown_task_data':
        return CountdownTasksDataset(**kwargs)
    elif dataset == 'gsm8k_task_data':
        return GSM8k(**kwargs)
    else:
        raise ValueError(f'Unimplemented dataset {dataset}')

def reward_function_dispatcher(dataset: str) -> Callable:
    if dataset=='countdown_task_data':
        return countdown_reward_function_dispatcher
    elif dataset == 'gsm8k_task_data':
        return gsm8k_reward_function_dispatcher