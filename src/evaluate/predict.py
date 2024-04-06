# write function to predict on new data
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk
import src.utils as utils
import torch
  

def predict_ner_from_dataset(dataset_path, text_feature_name, model_output_path):
    """
    this dataset_path could be a csv or a json file with a text_feature_name column
    """
    dataset = utils.read_file(dataset_path)
    text_list = [row[text_feature_name] for row in dataset]
    model = AutoModelForTokenClassification.from_pretrained(model_output_path)
    tokenizer = utils.get_model_tokenizer()
  
    tokenised_outputs = predict_ner_for_text_batch(text_list=text_list, model=model, tokenizer=tokenizer)
    return tokenised_outputs


def predict_ner_from_hf_dataset(dataset_hf, model, batch_size=8):
    model.eval()  # Ensure model is in evaluation mode

    # Initialize a container for aggregated results, dynamically including original dataset features
    aggregated_results = {feature: dataset_hf[feature] for feature in dataset_hf.features.keys()}
    # Add new keys for prediction outputs
    aggregated_results.update({
        'prediction_probabilities': [],
        'prediction_labels': [],
        'bio_tags': [],
    })

    for i in range(0, len(dataset_hf), batch_size):
        # Extract batch for all features
        batch = {k: dataset_hf[k][i: i+batch_size] for k in dataset_hf.features.keys()}
        input_ids = torch.tensor(batch['input_ids'])
        attention_mask = torch.tensor(batch['attention_mask'])

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = logits.argmax(-1)
        bio_tags = [[model.config.id2label[label.item()] for label in prediction] for prediction in predictions]

        # Store new prediction outputs
        aggregated_results['prediction_probabilities'].append(logits)
        aggregated_results['prediction_labels'].append(predictions)
        aggregated_results['bio_tags'].extend(bio_tags)

    # Post-process to concatenate tensors where applicable
    for feature, data_list in aggregated_results.items():
        if feature in ['prediction_probabilities', 'prediction_labels']:
            aggregated_results[feature] = torch.cat(data_list, dim=0)
        # Other features are kept as lists or processed according to specific needs

    return aggregated_results


    

def predict_ner_for_text(text_list, model, tokenizer):
    """
    This function returns a dictionary with the following keys:
    - input_ids
    - attention_mask
    - token_type_ids
    - outputs
    - overflow_to_sample_mapping
    """
    tokenized_inputs = utils.tokenize_text_inputs(text_list, tokenizer)
    
    # Prediction
    outputs = model(torch.tensor(tokenized_inputs['input_ids']), torch.tensor(tokenized_inputs['attention_mask']))["logits"]
    # add outputs to tokenized_inputs
    tokenized_inputs['prediction_probabilities'] = outputs
    tokenized_inputs['prediction_labels'] = outputs.argmax(-1)
    tokenized_inputs['bio_tags'] = [[model.config.id2label[label] for label in labels_per_ex] for labels_per_ex in tokenized_inputs['prediction_labels'].tolist() ]
    
    return tokenized_inputs


def predict_ner_for_text_batch(text_list, model, tokenizer, batch_size=8):
    """
    This function returns a dictionary with the following keys:
    - input_ids
    - attention_mask
    - token_type_ids
    - outputs
    - overflow_to_sample_mapping
    """
    
    tokenized_inputs = {}
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i: min(i+batch_size, len(text_list))]
        
        # this implementation pads to 512 tokens and not memory optimised
        batch_tokenized_inputs = utils.tokenize_text_inputs(batch_text, tokenizer)

        outputs = model(torch.tensor(batch_tokenized_inputs['input_ids']),
                        torch.tensor(batch_tokenized_inputs['attention_mask']))["logits"]
        batch_tokenized_inputs['prediction_probabilities'] = outputs
        batch_tokenized_inputs['prediction_labels'] = outputs.argmax(-1)
        batch_tokenized_inputs['bio_tags'] = [[model.config.id2label[label.item()] for label in labels_per_ex] for labels_per_ex in batch_tokenized_inputs['prediction_labels']]
        
        # Convert 'bio_tags' list to tensor if necessary. Here, you might need a custom approach
        # as 'bio_tags' are categorical and variable in length. Consider keeping them as lists or
        # encode them differently if a tensor representation is required.

        # For other keys, concatenate the tensors
        for key in batch_tokenized_inputs:
            if key == 'bio_tags':
                if tokenized_inputs.get(key) is None:
                    tokenized_inputs[key] = batch_tokenized_inputs[key]
                else:
                    tokenized_inputs[key] += batch_tokenized_inputs[key]
                continue
            batch_tensor = torch.tensor(batch_tokenized_inputs[key]) 
            if tokenized_inputs.get(key) is None:
                tokenized_inputs[key] = batch_tensor
            else:
                tokenized_inputs[key] = torch.cat((tokenized_inputs[key], batch_tensor), dim=0)
    
    return tokenized_inputs


def process_outputs(tokenized_outputs):
    """
    Process the model outputs to extract named entities in a JSON format.

    :param tokenized_outputs: Tokenized outputs from the model.
    :param tokenizer: The tokenizer used for tokenizing input texts.
    :return: A JSON-like list of dictionaries containing extracted entities.
    
    """

    overflow_to_sample_mapping = tokenized_outputs["overflow_to_sample_mapping"]

    # Convert model predictions to label names
    predicted_labels = tokenized_outputs["bio_tags"]
    offset_mapping = tokenized_outputs["offset_mapping"]#.detach().to_numpy()
    
    extracted_entities = {}
    
    for i, labels in enumerate(predicted_labels):
        
        sample_index = overflow_to_sample_mapping[i]
        if sample_index not in extracted_entities:
            extracted_entities[sample_index] = []
            
        offsets = offset_mapping[i]
        entity = None
        for label, token_offset in zip(labels, offsets):
            if label.startswith("B-"):
                if entity:
                    extracted_entities[sample_index].append(entity)  # Add the previous entity in the list if the previous word was part of an entity
                entity = {"prediction_label": label[2:], "start": token_offset[0], "end": token_offset[1]}
            elif label.startswith("I-") and entity and label[2:] == entity["prediction_label"]:    
                entity["end"] = token_offset[1]
            elif label == 'O' or (entity and label[2:] != entity["prediction_label"]):
                if entity:
                    extracted_entities[sample_index].append(entity)
                    entity = None

                    
        ### check error
        # if entity and len(extracted_entities)==i+1:  # Add the last entity if it hasn't been added
        #     extracted_entities[i].append(entity)
    return extracted_entities


