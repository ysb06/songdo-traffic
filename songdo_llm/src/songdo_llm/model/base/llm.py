import torch
import torch.nn as nn
import logging
from typing import Dict, List
import os

from ollama import chat, ChatResponse
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer
)
from transformers.modeling_outputs import SequenceClassifierOutput
import yaml


class LLMWeight(nn.Module):
    def __init__(
        self,
        llm_name: str = "deepseek-r1:8b",
        classifier_name: str = "microsoft/deberta-v3-base",
        llm_cache_path: str = "./output/llm_cache",
    ):
        super(LLMWeight, self).__init__()
        self.llm_name = llm_name
        self.tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(
            classifier_name
        )
        self.cls_model = DebertaV2ForSequenceClassification.from_pretrained(classifier_name, num_labels=1)        
        self.llm_str_cache: Dict[str, str] = {}
        self.llm_tkn_cache: Dict[str, List[int]] = {}

        self.llm_str_cache_path = os.path.join(
            llm_cache_path, "_str_", f"{llm_name.replace('[^a-zA-Z0-9]', '_')}.yaml"
        )
        if os.path.exists(self.llm_str_cache_path):
            with open(self.llm_str_cache_path, "r") as f:
                self.llm_str_cache = yaml.safe_load(f)
        for key, value in self.llm_str_cache.items():
            self.llm_tkn_cache[key] = self.tokenizer.encode(value, return_tensors="pt")

    def forward(self, x: pd.Timestamp) -> torch.Tensor:
        query_key = x.strftime("%B %d")
        if query_key not in self.llm_str_cache:
            query_text = f"Answer anything related to all of the following: {query_key}; Korea; and traffic."
            response: ChatResponse = chat(
                model="llama3.1",
                messages=[{"role": "user", "content": query_text}],
            )
            response_text = response.message.content
            self.llm_str_cache[query_key] = response_text
            self.llm_tkn_cache[query_key] = self.tokenizer(response_text, return_tensors="pt")

        y_2_hat: SequenceClassifierOutput = self.cls_model(**self.llm_tkn_cache[query_key])

        return y_2_hat.logits
