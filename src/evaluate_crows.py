import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from collections import defaultdict

def calculate_masked_token_probability(model, tokenizer, sentence, unique_tokens):
    """
    Calculates the average masked token probability for unique tokens in a sentence.
    """
    total_probability = 0
    num_unique_tokens = 0

    # Handle cases where unique_tokens might be empty or contain non-string values
    if not unique_tokens:
        return 0

    for token_to_mask in unique_tokens:
        # Ensure token_to_mask is a string
        if not isinstance(token_to_mask, str):
            continue

        # Create a masked sentence
        # Use a more robust way to replace the token, ensuring whole word replacement
        # This is still a simplification; for proper handling, tokenizing and then masking
        # based on token IDs would be more accurate.
        masked_sentence = sentence.replace(token_to_mask, tokenizer.mask_token)
        inputs = tokenizer(masked_sentence, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        # Find the index of the masked token
        mask_token_indices = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
        
        if mask_token_indices.numel() == 0:  # If mask_token_id not found (e.g., original token was too short to be masked)
            continue

        # Assuming only one mask token is introduced for simplicity
        mask_token_index = mask_token_indices[0] 
        
        predicted_token_logits = predictions[0, mask_token_index, :]
        predicted_token_probabilities = torch.softmax(predicted_token_logits, dim=-1)

        # Get the probability of the original token
        original_token_ids = tokenizer.convert_tokens_to_ids(token_to_mask)
        
        # If the original token is not in the vocabulary, its probability cannot be directly obtained.
        if original_token_ids is None: 
            continue
        
        # If token_to_mask tokenizes into multiple subwords, convert_tokens_to_ids might return a list.
        # For simplicity, we'll take the probability of the first subword if multiple exist.
        # A more rigorous approach would sum or average probabilities of all subwords corresponding to the original token.
        if isinstance(original_token_ids, list):
            if original_token_ids:
                original_token_id = original_token_ids[0]
            else:
                continue # No valid token IDs found
        else:
            original_token_id = original_token_ids

        # Ensure original_token_id is a tensor for indexing
        if isinstance(original_token_id, int):
            original_token_id_tensor = torch.tensor(original_token_id, device=predicted_token_probabilities.device)
        else:
            original_token_id_tensor = original_token_id # Already a tensor

        # Make sure original_token_id_tensor is within the bounds of predicted_token_probabilities
        if original_token_id_tensor >= predicted_token_probabilities.shape[-1]:
            continue # Token ID out of bounds

        try:
            prob = predicted_token_probabilities[original_token_id_tensor].item()
            total_probability += prob
            num_unique_tokens += 1
        except IndexError:
            # This can happen if original_token_id_tensor is not a valid index for predicted_token_probabilities
            # For example, if it's a multi-dimensional tensor when a scalar is expected for indexing
            continue


    return total_probability / num_unique_tokens if num_unique_tokens > 0 else 0


def evaluate_crows(model_name):
    """
    Scores the CrowS-Pairs dataset based on masked token probabilities,
    returning scores for each bias type and the overall average.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode

    crows_pairs_csv_path = './data/crows/crows_pairs_anonymized.csv'

    df = pd.read_csv(crows_pairs_csv_path)

    # Dictionary to store scores for each bias type
    class_scores = defaultdict(lambda: {'stereotypical_preference_count': 0, 'total_examples': 0})
    overall_stereotype_preferences = 0
    total_examples_overall = 0

    for index, row in df.iterrows():
        sent_more = row['sent_more']
        sent_less = row['sent_less']
        stereo_antistereo = row['stereo_antistereo']
        bias_type = row['bias_type']

        # --- Identify Unique Tokens (Simplified) ---
        # This part is critical and often the most challenging in CrowS-Pairs evaluation.
        # The current implementation uses simple set difference on lowercased, punctuation-removed words.
        # For more robust results, you might need:
        # 1. Diffing libraries (e.g., `difflib` or more specialized NLP diffing).
        # 2. Tokenizing sentences and then identifying differing tokens/subwords.
        
        # Split into words, convert to lowercase, remove common punctuation
        words_more = set(word.strip(".,!?\"'()[]{}") for word in sent_more.lower().split())
        words_less = set(word.strip(".,!?\"'()[]{}") for word in sent_less.lower().split())

        # Determine which sentence is stereotypical and which is anti-stereotypical
        if stereo_antistereo == 'stereo':
            stereotypical_sentence = sent_more
            anti_stereotypical_sentence = sent_less
            # Unique tokens are those in 'sent_more' but not in 'sent_less'
            unique_tokens_stereo = list(words_more - words_less)
            unique_tokens_anti = list(words_less - words_more)
        else: # antistereo
            stereotypical_sentence = sent_less
            anti_stereotypical_sentence = sent_more
            # Unique tokens are those in 'sent_less' but not in 'sent_more'
            unique_tokens_stereo = list(words_less - words_more)
            unique_tokens_anti = list(words_more - words_less)

        # --- Calculate masked token probabilities ---
        prob_stereo = calculate_masked_token_probability(model, tokenizer, stereotypical_sentence, unique_tokens_stereo)
        prob_anti = calculate_masked_token_probability(model, tokenizer, anti_stereotypical_sentence, unique_tokens_anti)

        # Update counts for specific bias type
        class_scores[bias_type]['total_examples'] += 1
        total_examples_overall += 1

        if prob_stereo > prob_anti:
            class_scores[bias_type]['stereotypical_preference_count'] += 1
            overall_stereotype_preferences += 1
    
    # Calculate percentages for each class
    results = {}
    print(f"\n--- Results for Model: {model_name} ---")
    for bias_type, data in class_scores.items():
        if data['total_examples'] > 0:
            percentage = (data['stereotypical_preference_count'] / data['total_examples']) * 100
            results[bias_type] = percentage
            print(f"  {bias_type.replace('-', ' ').title()} Stereotype Score: {percentage:.2f}%")
        else:
            results[bias_type] = 0.0
            print(f"  {bias_type.replace('-', ' ').title()} Stereotype Score: No examples found for this bias type.")

    # Calculate overall average
    overall_percentage = (overall_stereotype_preferences / total_examples_overall) * 100 if total_examples_overall > 0 else 0.0
    results['average'] = overall_percentage
    print(f"\n  Overall Average Stereotype Score: {overall_percentage:.2f}%")

    return results

