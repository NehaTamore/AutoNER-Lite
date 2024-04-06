import json
from typing import List
from src.utils import get_project_config
project_config = get_project_config()
    
ENTITY_DICT = project_config["annotation_settings"]["entities"]
LLM_ENTITY_CONERSION_DICT = dict({value: entity_info["type"] for entity_info in ENTITY_DICT for value in entity_info["synon_names"]})

ENTITY_TYPES = set({entity_info["type"] for entity_info in ENTITY_DICT})


def standardise_llm_entity_tpes(data: List) -> List:
    """
    LLMs tend to sometimes use different entity type names for the same thing. This function standardises the entity types.
    """
    for example in data:
        for entity in example['extracted_entities']:
            entity['label'] = LLM_ENTITY_CONERSION_DICT.get(entity['label'], entity['label'])
                      
    return data


def filter_llm_entity_types(data: List, entity_types: List) -> List:
    """
    Filters out all entities that are not in the entity_types list.
    """
    for example in data:
        example['extracted_entities'] = [entity for entity in example['extracted_entities'] if entity['label'] in entity_types]
        
    return data


def correct_llm_response(input_file_path, output_file_path):
    """
    Corrects the LLM response by standardising the entity types and filtering out unwanted entity types.
    """
    with open(input_file_path, ) as f:
        data = json.load(f)
        
    data = standardise_llm_entity_tpes(data)
    data = filter_llm_entity_types(data, ENTITY_TYPES)
    
    # Save the corrected data
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    return
