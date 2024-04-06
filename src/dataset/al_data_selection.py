import src.utils as utils
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.predict import predict_ner_for_text_batch


tokenizer = utils.get_model_tokenizer()


def get_top_k_example_indices(scores, k):
    # sort the scores to get get indices and values of top k scores
    _, sorted_indices = torch.sort(torch.from_numpy(scores), dim=-1, descending=False)
    return sorted_indices[:k]
    
    
def margin_sampling(predictions, attention_mask, top_probs):
    
    top_prob_indices = np.argsort(-predictions, axis=-1)[..., :top_probs]
    # print(top_prob_indices[0])
    
    batch_indices = np.arange(predictions.shape[0])[:, np.newaxis, np.newaxis]
    seq_indices = np.arange(predictions.shape[1])[np.newaxis, :, np.newaxis]
    top_probs_values = predictions[batch_indices, seq_indices, top_prob_indices]
    
    # Compute Modified Margins
    # Calculate margins between the 1st and each of the next top_probs - 1 probabilities
    first_prob = top_probs_values[..., 0:1]
    other_probs = top_probs_values[..., 1:]
    margins = first_prob - other_probs

   
    avg_margins = np.mean(margins, axis=-1)

    
    avg_margins *= attention_mask
    
    # Sum across tokens
    avg_margins = np.sum(avg_margins, axis=-1)
    print(avg_margins.shape)
    return avg_margins


def aggregate_and_normalize_scores(scores, overflow_to_sample_mapping, attention_mask):
    aggregated_scores = np.zeros(max(overflow_to_sample_mapping) + 1)
    token_counts = np.zeros_like(aggregated_scores)
    print(aggregated_scores.shape)
    for score, mapping, mask in zip(scores, overflow_to_sample_mapping, attention_mask):
        # Sum up scores for the same original input
        aggregated_scores[mapping] += score
        # Count tokens for the same original input, excluding padding ([PAD] tokens)
        token_counts[mapping] += np.sum(mask)
    
    # Normalize scores by the token count for each original input
    normalized_scores = aggregated_scores / token_counts
    return normalized_scores

def adjust_overflow_to_sample_mapping(overflow_to_sample_mapping):
    overflow_to_sample_mapping_new = [0]
    for index in range(len(overflow_to_sample_mapping)):
        if index==0:
            continue
        if overflow_to_sample_mapping[index]==overflow_to_sample_mapping[index-1]:
            overflow_to_sample_mapping_new.append(overflow_to_sample_mapping_new[index-1])
        else:
            overflow_to_sample_mapping_new.append(max(overflow_to_sample_mapping_new)+1)
            
    return torch.tensor(overflow_to_sample_mapping_new)
              

def entropy_based_sampling(predictions, attention_mask, top_probs):
    # Step 1: Calculate Top k Probabilities
    top_prob_indices = np.argsort(-predictions, axis=-1)[..., :top_probs]
    
    # Gather the top probabilities
    batch_indices = np.arange(predictions.shape[0])[:, np.newaxis, np.newaxis]
    seq_indices = np.arange(predictions.shape[1])[np.newaxis, :, np.newaxis]
    top_probs_values = predictions[batch_indices, seq_indices, top_prob_indices]

    # Step 2: Normalize these Probabilities
    # We need to normalize these probabilities so that they sum up to 1
    normalized_top_probs = top_probs_values / np.sum(top_probs_values, axis=-1, keepdims=True)

    # Step 3: Calculate Entropy
    # Using the normalized probabilities, calculate entropy
    entropy = -np.sum(normalized_top_probs * np.log(normalized_top_probs + 1e-9), axis=-1)  # Adding a small constant to avoid log(0)

    # Step 4: Apply Attention Mask
    # Use the attention mask to filter out entropy values for padding tokens
    entropy *= attention_mask
    
    return entropy


def filter_tagged_examples(df):
    return df.loc[df["al_iteration_num"]==-1]


def select_examples_with_al(trained_model_file_path,
                            file_name, 
                            id_column_name,
                            text_column_name,
                            top_probs,
                            sampling_strategy,
                            num_examples):
    df = pd.read_csv(file_name) 
    df = df.dropna(subset=[id_column_name, text_column_name])
    # df = filter_tagged_examples(df)   ########################################### remove this line
    cleaned_text_list = df.loc[:, text_column_name].tolist()
    
    ids_list = df[id_column_name].tolist()
    
    if sampling_strategy == "random":
        # randomly select 10 example ids_list
        return np.random.choice(ids_list, num_examples)

    model = AutoModelForTokenClassification.from_pretrained(trained_model_file_path)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_file_path)
    tokenised_inputs = predict_ner_for_text_batch(cleaned_text_list, model, tokenizer, batch_size=4)
    
    predictions = tokenised_inputs['prediction_probabilities'].clone().detach().numpy()
    attention_mask = tokenised_inputs['attention_mask'].clone().detach().numpy()
    overflow_to_sample_mapping = tokenised_inputs['overflow_to_sample_mapping']
    overflow_to_sample_mapping = adjust_overflow_to_sample_mapping(overflow_to_sample_mapping)
    if sampling_strategy == 'uncertainty_sampling':
        al_scores = margin_sampling(predictions, attention_mask, top_probs)
    elif sampling_strategy == 'entropy_sampling':
        al_scores = entropy_based_sampling(predictions, attention_mask, top_probs)
    else:
        raise ValueError("Sampling strategy must be either 'uncertainty' or 'entropy'")

    # Aggregate and normalize uncertainty scores based on token count
    normalized_scores = aggregate_and_normalize_scores(al_scores, overflow_to_sample_mapping, attention_mask)
    print(normalized_scores.shape)
    # Assuming get_top_k_example_indices and other necessary functions are defined elsewhere
    indices = get_top_k_example_indices(normalized_scores, k=num_examples)
    ids = [ids_list[i] for i in indices]

    return ids
