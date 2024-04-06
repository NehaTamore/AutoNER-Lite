from src.utils import get_model_tokenizer, tokenize_text_inputs
from datasets import Dataset
from src.utils import get_project_config, read_json_file
project_config = get_project_config()
SPECIAL_TOKEN = project_config["llm_to_hf_dataset_preprocessor_settings"]["entity_encoding"]["special_token_tag"]


def find_last_non_zero_index(token_offsets):
    """
    Finds the last non-zero index in the token_offsets list, 
    before the given index i.
    """
    for j in range(len(token_offsets)):
        if token_offsets[j] != (0, 0):
            return token_offsets[j][-1]
    return 0


def create_bio_tags(json_data, tokenized_inputs, tokenizer):
    bio_tags = []

    # Extract token offsets and overflow mapping
    token_offsets = tokenized_inputs['offset_mapping']
    overflow_to_sample_mapping = tokenized_inputs['overflow_to_sample_mapping']
    
    for i in range(len(tokenized_inputs["input_ids"])):
        # Find the index of the original sample to which the current token belongs
        sample_index = overflow_to_sample_mapping[i]
        entities = json_data[sample_index]["extracted_entities"]

        # Sort entities based on start index
        entities = sorted(entities, key=lambda x: x['start'])

        input_offset = token_offsets[i]

        sample_tags = []
        
        for j, (token_start, token_end) in enumerate(input_offset):
            token_start = token_start 
            token_end = token_end 
            
            # Special tokens (CLS, SEP, PAD) have a specific offset pattern (0, 0)
            if (token_start, token_end) == (0, 0):
                token_tag = SPECIAL_TOKEN  # Assign 'Outside' tag to special tokens
            else:
                token_tag = "O"  # Default tag is 'Outside'

                for entity in entities:
                    # Check if the token is inside an entity
                    if token_start >= entity['start'] and token_end <= entity['end']:
                        if token_start == entity['start']:
                            token_tag = f"B-{entity['label']}"  # Begin tag
                        else:
                            token_tag = f"I-{entity['label']}"  # Inside tag
                        break
            
            sample_tags.append(token_tag)
        
        bio_tags.append(sample_tags)
    tokenized_inputs['bio_tags'] = bio_tags    
    
    return tokenized_inputs


def map_tags_to_labels(dataset):
    project_config = get_project_config()
    entity_mapping_path = project_config["data_paths"]["entities_to_label_mapping_path"]
    print(entity_mapping_path)
    entity_mapping = read_json_file(entity_mapping_path)
    label2id = entity_mapping["label2id"]
    
    dataset = dataset.map(lambda x: {'labels': [label2id[label] for label in x['bio_tags']]})
    return dataset


def create_dataset(json_data_path, data_output_path):
    """
    Data is the json data
    """
    tokenizer = get_model_tokenizer()
    json_data = read_json_file(json_data_path)
    concatenated_input_text_list = [example['concatenated_input_text'] for example in json_data]

    tokenized_outputs = tokenize_text_inputs(concatenated_input_text_list, tokenizer)

    ids = [example['id'] for example in json_data]
    ids_with_overflow = [ids[overflow_mapping] for overflow_mapping in tokenized_outputs['overflow_to_sample_mapping']]
    
    tokenized_outputs['id'] = ids_with_overflow

    tokenized_outputs_with_tags = create_bio_tags(json_data, tokenized_outputs, tokenizer)
    dataset = Dataset.from_dict(tokenized_outputs_with_tags)
    dataset = map_tags_to_labels(dataset)

    dataset.save_to_disk(data_output_path)
    
    return
