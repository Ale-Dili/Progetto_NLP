import pandas as pd
import torch
from tqdm import tqdm
import csv
import numpy as np
import difflib
import torch.nn.functional as F
from math import log


def evaluate_crows(model, tokenizer, csv_path='./data/crows/crows_pairs_anonymized.csv', device='mps'):
    
    def masked_token_prob(tokens, token_idx, target_token):
        '''
        Compute probability of a masked token in a sentence a
            - tokens: tokenized sentence
            - token_idx: index of the target_token among the tokens (tokenized sentence)
            - target_token: token to mask
        '''
        tokens = tokens[:] #shallow copy to avoid edit by reference
        tokens[token_idx] = tokenizer.mask_token
        enc = tokenizer.encode_plus(
                tokens,
                is_split_into_words=True,
                return_tensors="pt"
            )
        enc.to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        #return the position of the masked token. token_idx != mask_pos due to tokenization
        mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item() 
        dist = F.softmax(logits[0, mask_pos], dim=-1)
        tgt_id = tokenizer.convert_tokens_to_ids(target_token) #token id of the target token. NOTE it is note the idx in the sentence
        return dist[tgt_id].item()


    model.to(device)
    model.eval()

    BIAS_TYPES = [
        "general",
        "race-color",
        "socioeconomic",
        "gender",
        "disability",
        "nationality",
        "sexual-orientation",
        "physical-appearance",
        "religion",
        "age",
    ]

    n_stereo = {b:0 for b in BIAS_TYPES}
    n_anti = {b:0 for b in BIAS_TYPES}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=1508): #1508 examples 
            direction = row["stereo_antistereo"]
            bias_type = row["bias_type"]

            
            if direction == "stereo":
                stereo_sent = row["sent_more"]
                anti_sent = row["sent_less"]
            else:
                stereo_sent = row["sent_less"]
                anti_sent = row["sent_more"]
            
            stereo_token = tokenizer.tokenize(stereo_sent)
            anti_token = tokenizer.tokenize(anti_sent)
            

            sm = difflib.SequenceMatcher(None, stereo_token, anti_token)
            stereo_unique, anti_unique = [], []
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ("replace", "delete"):
                    stereo_unique.extend(stereo_token[i1:i2])
                if tag in ("replace", "insert"):
                    anti_unique.extend(anti_token[j1:j2])

            if len(stereo_unique)==0 or len(anti_unique)==0:
                continue

            #likelihood stereo
            stereo_log_likelihood = 0
            for idx,s_t in enumerate(stereo_token):
                if s_t not in stereo_unique:
                    stereo_log_likelihood += log(masked_token_prob(stereo_token,idx,s_t))
            
            #likelihood anti
            anti_log_likelihood = 0
            for idx,a_t in enumerate(anti_token):
                if a_t not in anti_unique:
                    anti_log_likelihood += log(masked_token_prob(anti_token,idx,a_t))

            if stereo_log_likelihood > anti_log_likelihood:
                n_stereo['general']+=1
                n_stereo[bias_type]+=1
            else:
                n_anti['general']+=1
                n_anti[bias_type]+=1

        result = {
            b: n_stereo[b]/(n_stereo[b]+n_anti[b])
            for b in BIAS_TYPES
        }    

        return result