import pandas as pd
import torch
from tqdm import tqdm

import numpy as np

import torch.nn.functional as F

from math import log
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)


#TODO handle more datasets
class PerformanceEvaluator:
    def __init__(self, model, tokenizer,device='mps'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()
        self.model.to(device)

        self.SEED = 44
        self.DS_NAME = 'stas/openwebtext-10k'
        self.SPLIT = "train"
        self.NUM_SAMPLES = 4000             
        self.BATCH_SIZE = 8
        MASK_PROB = 0.15

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=MASK_PROB
        ) 


    def _tokenize_batch(self,batch):
        return self.tokenizer(batch["text"], truncation=True, max_length=128)
    

    def masked_token_evaluation(self):
        ds = load_dataset(self.DS_NAME, split=self.SPLIT)
        ds = ds.shuffle(seed=self.SEED).select(range(self.NUM_SAMPLES))

        tokenized = ds.map(self._tokenize_batch, batched=True, remove_columns=["text"])
        loader = DataLoader(tokenized, batch_size=self.BATCH_SIZE, collate_fn=self.collator)
        
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels    = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits  # (B, L, V)

                preds = logits.argmax(dim=-1)  # (B, L)
                mask = labels != -100

                correct += (preds[mask] == labels[mask]).sum().item()
                total   += mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return accuracy



