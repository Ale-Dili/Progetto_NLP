import pandas as pd
import torch
from tqdm import tqdm

import numpy as np

import torch.nn.functional as F

from math import log
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from sklearn.metrics import precision_recall_fscore_support


class PerformanceEvaluator:
    def __init__(self, model, tokenizer,device='mps'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(device)

        self.SEED = 44
        


    def _tokenize_batch(self,batch):
        return self.tokenizer(batch["text"], truncation=True, max_length=128)
    

    def masked_token_evaluation(self, K: int = 5):
        DS_NAME = 'stas/openwebtext-10k'
        SPLIT = "train"
        NUM_SAMPLES = 4000
        BATCH_SIZE = 8
        MASK_PROB = 0.15

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=MASK_PROB
        ) 

        ds = load_dataset(DS_NAME, split=SPLIT)
        ds = ds.shuffle(seed=self.SEED).select(range(NUM_SAMPLES))

        tokenized = ds.map(self._tokenize_batch, batched=True, remove_columns=["text"])
        loader = DataLoader(tokenized, batch_size=BATCH_SIZE, collate_fn=collator)
        
        correct, total = 0, 0
        correct_topk = 0

        with torch.no_grad():
            for batch in tqdm(loader):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits  # (B, L, V)

                # standard accuracy
                preds = logits.argmax(dim=-1)  # (B, L)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()

                #top-k accuracy
                topk_inds = logits.topk(K, dim=-1).indices  # (B, L, K)

                #for each masked position, check if label in top-k
                correct_topk += (
                    topk_inds[mask.unsqueeze(-1).expand_as(topk_inds)]  # (sum(mask), K)
                    .eq(labels[mask].unsqueeze(-1))                  # (sum(mask), K) bool
                    .any(dim=-1)                                     # (sum(mask),)     bool
                    .sum()
                    .item()
                )

                total += mask.sum().item()

        accuracy     = correct / total if total > 0 else 0.0
        topk_accuracy = correct_topk / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            f"top_{K}_accuracy": topk_accuracy
        }




    def text_classification_evaluation(self):
        DATASET_NAME = "ag_news"
        SPLIT = "test"
        NUM_SAMPLES = 1000

        class_to_word = {
            0: "world",
            1: "sports",
            2: "business",
            3: "technology"
        }

        dataset = load_dataset(DATASET_NAME, split=SPLIT).select(range(NUM_SAMPLES))

        preds = []
        labels = []

        for sample in tqdm(dataset):
            prompt = f"{sample['text']}. {self.tokenizer.sep_token} The news topic is {self.tokenizer.mask_token}."
            inputs = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                mask_positions = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                if len(mask_positions) == 0:
                    continue

                mask_pos = mask_positions[0].item()
                mask_logits = logits[0, mask_pos, :]


                scores = {}
                for class_idx, word in class_to_word.items():
                    token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
                    scores[class_idx] = mask_logits[token_id].item()

                pred = max(scores.items(), key=lambda x: x[1])[0]

            preds.append(pred)
            labels.append(sample['label'])

        #macro
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )


        return {
            "accuracy": sum(1 for p,l in zip(preds, labels) if p==l)/len(labels),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }



