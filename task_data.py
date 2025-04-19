from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from tokenizer import Tokenizer

class Task_Dataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        data_sub_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / data_sub_path)
                # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError('Please subclass Task_dataset and implement __getitem__')

    def encode_prefix(self, question: str):
        raise NotImplementedError('Please subclass Task_dataset and implement encode_prefix')
    
