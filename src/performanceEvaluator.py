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
        


    def _tokenize_batch(self,batch):
        return self.tokenizer(batch["text"], truncation=True, max_length=128)
    

    def masked_token_evaluation(self):
        DS_NAME = 'stas/openwebtext-10k'
        SPLIT = "train" #only split available
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
        correct = 0
        for sample in tqdm(dataset):
            prompt = f"{sample['text']}. {self.tokenizer.sep_token} The news topic is {self.tokenizer.mask_token}."
            inputs = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                if len(mask_token_index) == 0:
                    continue
            
                mask_token_logits = logits[0, mask_token_index[0], :]
                
                probs = {}
                for class_idx, word in class_to_word.items():
                    word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
                    probs[class_idx] = mask_token_logits[word_id].item()
                
                pred = max(probs.items(), key=lambda x: x[1])[0]
                
            if pred == sample['label']:
                correct += 1

        accuracy = correct / NUM_SAMPLES
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy
        

#TODO finish
    def sentiment_analysis_evaluation(self):
        ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", split='test')
        class_to_word = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }

        correct = 0
        total = 0

        for sample in tqdm(ds):
            #prompt = f"{sample['text']}. {self.tokenizer.sep_token} The sentiment is {self.tokenizer.mask_token}."
            prompt = f"Review: {sample['text']} Sentiment: {self.tokenizer.mask_token}."
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
     

            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            if len(mask_token_index) == 0:
                continue
            
            p = F.softmax(logits[0])[mask_token_index][0]

            probs = {}
            for class_idx, word in class_to_word.items():
                word_id = self.tokenizer.convert_tokens_to_ids(word)
                probs[word] = p[word_id]

            pred = max(probs.items(), key=lambda x: x[1])[0]


            if pred == sample['sentiment']:
                correct += 1
            total += 1

        accuracy = correct / total 
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy
