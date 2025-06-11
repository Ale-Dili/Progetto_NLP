import pandas as pd
import torch
from tqdm import tqdm


#compute pseudo log likelihood while masking token
def score_sentence(sentence, model, tokenizer, device='mps'):
    model.to(device)
    model.eval()

    enc = tokenizer(sentence, return_tensors='pt')
    input_ids = enc['input_ids'][0]
    mask_id = tokenizer.mask_token_id #should generalize for each model

    total_log_prob = 0.0
    #except cls and sep tokens
    for i in range(1, len(input_ids) - 1):
        orig_id = input_ids[i].item()
        masked_ids = input_ids.clone()
        masked_ids[i] = mask_id

        masked_ids = masked_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(masked_ids)

        with torch.no_grad():
            outputs = model(masked_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape [1, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits[0, i], dim=-1)
        total_log_prob += log_probs[orig_id].item()

    return total_log_prob


def evaluate_crows(model, tokenizer, csv_path='./data/crows/crows_pairs_anonymized.csv'):

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    df = pd.read_csv(csv_path)

    scores_more, scores_less, preds = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        s_more = score_sentence(row['sent_more'], model, tokenizer, device)
        s_less = score_sentence(row['sent_less'], model, tokenizer, device)
        scores_more.append(s_more)
        scores_less.append(s_less)
        preds.append('stereo' if s_more > s_less else 'antistereo')

    df['score_more'] = scores_more
    df['score_less'] = scores_less
    df['prediction'] = preds

    results = {}
    overall_correct = 0
    overall_total = len(df)
    for cat, group in df.groupby('bias_type'):
        correct = (group['prediction'] == group['stereo_antistereo']).sum()
        total = len(group)
        results[cat] = correct / total
        overall_correct += correct
    results['overall'] = overall_correct / overall_total

    return results

