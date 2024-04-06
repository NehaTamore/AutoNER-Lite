from bs4 import BeautifulSoup
import json
from transformers import AutoTokenizer

def split_words(text):
    return text.split()

PROJECT_CONFIG_PATH = "project_config.json"


def clean_text_before_llm_tagging(text):
    if isinstance(text, str):
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator='. ').strip()
        words = split_words(text)
        text = ' '.join(words)
    return text


def get_project_config(project_config_path=PROJECT_CONFIG_PATH):
    with open(project_config_path) as f:
        project_config = json.load(f)
        return project_config


def read_json_file(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
        return data


def get_model_tokenizer(tokenizer_settings=None):
    if tokenizer_settings is None:
        project_config = get_project_config()
        tokenizer_settings = project_config["tokenizer_settings"]

    # Load the tokenizer from the specified model name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_settings["model_name"])

    # Dynamically apply settings from the dictionary
    for setting_name, setting_value in tokenizer_settings.items():
        if hasattr(tokenizer, setting_name):
            setattr(tokenizer, setting_name, setting_value)  # Set the attribute using setattr
        else:
            print(f"Warning: Tokenizer setting '{setting_name}' not found. Skipping.")

    return tokenizer


def tokenize_text_inputs(text_data, tokenizer):
    """Tokenizes a list of text inputs using a specified tokenizer.

    Args:
        text_data: A list of text strings to be tokenized.
        tokenizer: The tokenizer object to use for tokenization.

    Returns:
        A dictionary containing the tokenized inputs, as well as additional
        information such as overflowing tokens and offset mappings.
    """

    tokenized_outputs = tokenizer(
        text_data,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        is_split_into_words=False,
        return_offsets_mapping=True
    )
    
    return tokenized_outputs



def read_file(file_path):
    # based on the file extentions read the file
    if file_path.endswith('.json'):
        data = utils.read_json_file(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path).to_dict('records')
    else:
        raise ValueError(f'File type not supported: {file_path}')
    return data