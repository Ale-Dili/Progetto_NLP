import pandas as pd
import torch
from tqdm import tqdm
import csv
import numpy as np
import difflib
import torch.nn.functional as F
import json
from math import log


class StereosetEvaluator:
    def __init__(self, model, tokenizer, json_path ,device='mps'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.json_path = json_path

        self.model.eval()
        self.model.to(device)

        self.BIAS_TYPES = [
            'general',
            'race',
            'gender',
            'profession',
            'religion'
        ]


    def _masked_token_prob(self,tokens, token_idx, target_token):
        '''
        Compute probability of a masked token in a sentence a
            - tokens: tokenized sentence
            - token_idx: index of the target_token among the tokens (tokenized sentence)
            - target_token: token to mask
        '''
        tokens = tokens[:] #shallow copy to avoid edit by reference
        tokens[token_idx] = self.tokenizer.mask_token
        enc = self.tokenizer.encode_plus(
                tokens,
                is_split_into_words=True,
                return_tensors="pt"
            )
        enc.to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        #return the position of the masked token. token_idx != mask_pos due to tokenization
        mask_pos = (enc["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item() 
        dist = F.softmax(logits[0, mask_pos], dim=-1)
        tgt_id = self.tokenizer.convert_tokens_to_ids(target_token) #token id of the target token. NOTE it is note the idx in the sentence
        return dist[tgt_id].item()
    

    

    def evaluate_intrasentence(self):
        with open(self.json_path) as f:
            ds = json.load(f)['data']['intrasentence']

        n_stereo = {b:0 for b in self.BIAS_TYPES}
        n_anti = {b:0 for b in self.BIAS_TYPES}

        count = 0

        for element in tqdm(ds):
            bias_type = element['bias_type']

            for entry in element['sentences']:
                if entry['gold_label'] == 'stereotype':
                    stereo_sent = entry['sentence']
                elif entry['gold_label'] == 'anti-stereotype':
                    anti_sent = entry['sentence']


            stereo_token = self.tokenizer.tokenize(stereo_sent)
            anti_token = self.tokenizer.tokenize(anti_sent)

            sm = difflib.SequenceMatcher(None, stereo_token, anti_token)
            stereo_unique, anti_unique = [], []
            stereo_unique_idx, anti_unique_idx = [], [] #memorize idx in the tokenized sentence of the target token
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ("replace", "delete"):
                    for idx in range(i1, i2):
                        stereo_unique.append(stereo_token[idx])
                        stereo_unique_idx.append(idx)
                if tag in ("replace", "insert"):
                    for idx in range(j1, j2):
                        anti_unique.append(anti_token[idx])
                        anti_unique_idx.append(idx)

            if len(stereo_unique)==0 or len(anti_unique)==0:
                count+=1
                continue

            #if multiple tokens we take the avg
            #likelihood stereo
            stereo_log_likelihood = 0
            for idx,s_t in zip(stereo_unique_idx,stereo_unique):
                stereo_log_likelihood += log(self._masked_token_prob(stereo_token,idx,s_t))
            stereo_log_likelihood /= len(stereo_unique) #avg in case of multipe token

            anti_log_likelihood = 0
            for idx,a_t in zip(anti_unique_idx,anti_unique):
                anti_log_likelihood += log(self._masked_token_prob(anti_token, idx, a_t ))
            anti_log_likelihood /= len(anti_unique)

            if stereo_log_likelihood > anti_log_likelihood:
                n_stereo['general']+=1
                n_stereo[bias_type]+=1
            else:
                n_anti['general']+=1
                n_anti[bias_type]+=1

        result = {
            b: n_stereo[b]/(n_stereo[b]+n_anti[b])
            for b in self.BIAS_TYPES
        }    

        return result


    
    def evaluate_intersentence(self):
        with open(self.json_path) as f:
            ds = json.load(f)['data']['intersentence']

        n_stereo = {b:0 for b in self.BIAS_TYPES}
        n_anti = {b:0 for b in self.BIAS_TYPES}

        count = 0

        for element in tqdm(ds):
            bias_type = element['bias_type']
            context = element['context']


            for entry in element['sentences']:
                if entry['gold_label'] == 'stereotype':
                    stereo_sent = context + self.tokenizer.sep_token + entry['sentence']
                elif entry['gold_label'] == 'anti-stereotype':
                    anti_sent = context + self.tokenizer.sep_token + entry['sentence']

            stereo_token = self.tokenizer.tokenize(stereo_sent)
            stereo_sep_idx = stereo_token.index(self.tokenizer.sep_token)

            anti_token = self.tokenizer.tokenize(anti_sent)
            anti_sep_idx = anti_token.index(self.tokenizer.sep_token)

            stereo_log_likelihood = 0
            for idx in range(stereo_sep_idx+1, len(stereo_token)):
                s_t = stereo_token[idx]
                stereo_log_likelihood += log(self._masked_token_prob(stereo_token,idx,s_t))
            stereo_log_likelihood /= len(stereo_token)

            anti_log_likelihood = 0
            for idx in range(anti_sep_idx+1, len(anti_token)):
                a_t = anti_token[idx]
                anti_log_likelihood += log(self._masked_token_prob(anti_token,idx,a_t))
            anti_log_likelihood /= len(anti_token)

            if stereo_log_likelihood > anti_log_likelihood:
                n_stereo['general']+=1
                n_stereo[bias_type]+=1
            else:
                n_anti['general']+=1
                n_anti[bias_type]+=1

        result = {
            b: n_stereo[b]/(n_stereo[b]+n_anti[b])
            for b in self.BIAS_TYPES
        }    

        return result


