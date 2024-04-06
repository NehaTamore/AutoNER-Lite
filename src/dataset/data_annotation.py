import pandas as pd
# from langchain.llms import Bedrock
from tqdm import tqdm
import jsonlines

from bs4 import BeautifulSoup
from src.prompts import prompts_dict
import datetime
import json
import os

from src.utils import *

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from requests.exceptions import ReadTimeout
import os

# claude = Bedrock(
#     model_id="anthropic.claude-v2",
#     region_name="us-east-1",
#     model_kwargs={"max_tokens_to_sample": 15000, "temperature": 0.1},
# )

# claude.predict("hu")


project_config = get_project_config()

INPUT_COL_NAME = project_config["input_data_fields"]["text"]
ID_COL_NAME_IN_DF = project_config["input_data_fields"]["id"]
BATCH_SIZE = 10 # write output after every BATCH_SIZE results
NUM_WORDS_FOR_LLM_PER_INPUT = 400

TMP_OUTPUT_JSON_DATA_PATH = "data/annotated/"
LLM_LOGGING_ENABLED = True
LLM_LOG_PATH = "data/llm_logs/llm_log_" 
ID_COL_NAME = "id"
DOMAIN = project_config["domain"]

LLM_CACHE_FILE_PATH = "global_ner/active_learning_module_medical/data/annotated/llm_data_annotation_all.json"
# Read the LLM cache file
if os.path.exists(LLM_CACHE_FILE_PATH):
    llm_cache = dict()
    with open(LLM_CACHE_FILE_PATH, "r") as file:
        llm_cache_raw = json.load(file)
        for row in llm_cache_raw:
            llm_cache[row["id"]] = row
else:
    llm_cache = dict()

# create directory named data/annotated/
os.makedirs(TMP_OUTPUT_JSON_DATA_PATH, exist_ok=True)
if LLM_LOGGING_ENABLED:
    os.makedirs(LLM_LOG_PATH, exist_ok=True)


def extract_llm_response_in_json(llm_json_output):
    """
    Extracts entities from LLM JSON string output and returns the sentences and a list of extracted entities.

    Args:
        llm_json_output (list): The LLM JSON output containing sentence-wise entities.

    Returns:
        tuple: A tuple containing the sentences (str) and a list of extracted entities (list of dicts).
            Each entity dictionary contains the following keys:
            - "label" (str): The label of the entity.
            - "start" (int): The start index of the entity in the sentences.
            - "end" (int): The end index of the entity in the sentences.
    """

    sentence_index = 0
    output = []
    sentences = ""
    for sentence_wise_ents in llm_json_output:
        sentence = sentence_wise_ents["SENTENCE"].strip()
        # here we convert each sentence to spacy tokenisation
        # sentence = ' '.join([token.text for token in nlp(sentence)])
        entities = sentence_wise_ents["ENTITIES"]
        sentences += sentence + "\n"

        for entity_name, entity_values in entities.items():
            for entity_value in entity_values:
                entity_value = entity_value.strip()
                # entity_value = ' '.join([token.text for token in nlp(entity_value)])
                index = sentence.lower().find(entity_value.lower())
                if index != -1:
                    ent_begin_index = index
                    ent_end_index = index + len(entity_value)

                    if (
                        entity_value.lower()
                        != sentences[
                            ent_begin_index
                            + sentence_index : ent_end_index
                            + sentence_index
                        ].lower()
                    ):
                        print("didnt find this in input. Details as follows")
                        print(sentence)

                        print(
                            ent_begin_index + sentence_index,
                            ent_end_index + sentence_index,
                        )
                        print("entity value: ", entity_value)
                        print(
                            "extracted value: ",
                            sentences[
                                ent_begin_index
                                + sentence_index : ent_end_index
                                + sentence_index
                            ],
                        )

                        print()

                    else:
                        output.append(
                        {
                            "label": entity_name,
                            "start": ent_begin_index + sentence_index,
                            "end": ent_end_index + sentence_index,
                        }
                    )

        sentence_index += len(sentence) + 1

    return sentences, output


def clean_text_before_llm_tagging(text: str) -> str:
    if isinstance(text, str):
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=". ").strip()
        # text = ' '.join(text.split(" "))

    return text


def clean_llm_response(llm_response):
    try:
        return json.loads(llm_response)
    except json.JSONDecodeError:
        try:
            # "Here are the extracted entities from the given job description:" removing such beginnings
            llm_response = "\n".join(llm_response.split("\n")[1:])
        except:
            return None

    return llm_response


def llm_predict_on_input_chunks(preprocessed_text):
    """
    this function takes in a preprocessed text and returns the llm response
    some of the entities falling on the boundary of the chunks might be missed
    """
    
    preprocessed_text_chunks = []
    input_word_list = preprocessed_text.split()
    for i in range(0, len(input_word_list), NUM_WORDS_FOR_LLM_PER_INPUT):
        preprocessed_text_chunks.append(
            " ".join(input_word_list[i : i + NUM_WORDS_FOR_LLM_PER_INPUT])
        )

    # call llm on each chunk
    llm_extracted_entities_json = []
    for preprocessed_text_chunk in preprocessed_text_chunks:
        prompt = prompts_dict[DOMAIN].replace("input_____", preprocessed_text_chunk)
        llm_response = call_llm(prompt)
        llm_response = clean_llm_response(llm_response)
        if llm_response:
            json_llm_response = json.loads(llm_response)
            llm_extracted_entities_json.extend(json_llm_response)
        else:
            print("llm response is None")

    return llm_extracted_entities_json


def log_llm_response(prompt, llm_response, time_taken, llm_log_path):
    """
    Appends prompt, llm_response in a jsonl file if the file is not present for the current date else creates a new file.
    """

    with jsonlines.open(
        f"{llm_log_path}{datetime.datetime.now().strftime('%Y-%m-%d')}.jsonl", mode="a"
    ) as writer:
        writer.write(
            {"prompt": prompt, "llm_response": llm_response, "time": time_taken}
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(ReadTimeout),
)
def call_llm(prompt):
    start_time = datetime.datetime.now()
    llm_response = claude.predict(prompt)
    end_time = datetime.datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    if LLM_LOGGING_ENABLED:
        log_llm_response(prompt, llm_response, time_taken, LLM_LOG_PATH)
    return llm_response


def process_data(example_data):
    preprocessed_text = clean_text_before_llm_tagging(example_data)
    
    
    try:
        llm_extracted_entities_json = llm_predict_on_input_chunks(preprocessed_text)
        concatenated_input_text, extracted_entities = extract_llm_response_in_json(
            llm_extracted_entities_json
        )
        if concatenated_input_text:
            return {
                "preprocessed_text": preprocessed_text,
                "llm_response": llm_extracted_entities_json,
                "concatenated_input_text": concatenated_input_text,
                "extracted_entities": extracted_entities,
            }
        else:
            return None

    except Exception as e:
        print("Exception in process_data: ", e)
        print(example_data)

        return None


def process_batch(batch_data):
    """
    batch_data: df with a column INPUT_COL_NAME, and any other columns passed shall be added as a metafield to the output

    """
    batch_results = []
    for _, example_data in batch_data.iterrows():
        if example_data["id"] in llm_cache.keys():
            batch_results.append(llm_cache[example_data["id"]])
            continue
        example_text = example_data[INPUT_COL_NAME]
        
        example_meta_data = example_data.drop(
            INPUT_COL_NAME
        ).to_dict()  # such as job_id, job_title, domain, etc.

        result = process_data(example_text)

        if result:
            final_result = {**example_meta_data, **result}
            batch_results.append(final_result)

    return batch_results


def write_consolidated_outputs(output_json_data_path):
    file_names = os.listdir(TMP_OUTPUT_JSON_DATA_PATH)
    num_datapoints = 0
    all_results = []

    for file_name in file_names:
        try:
            with open(
                f"{TMP_OUTPUT_JSON_DATA_PATH}{file_name}", mode="r", encoding="latin-1"
            ) as file:
                batch_results = json.load(file)
                num_datapoints += len(batch_results)
                all_results.extend(batch_results)
        except json.JSONDecodeError:
            print(file_name)
            continue
    
    with open(f"{output_json_data_path}", mode="w") as file:
        json.dump(all_results, file)
        
    print("Total DataPoints annotated: ", num_datapoints)


def read_and_validate_input_data(input_data_path: str) -> pd.DataFrame:
    """
    input_data_path: path to the input data
    id column name in the input data should be "id"
    """
    data_df = pd.read_csv(input_data_path)
    
    if ID_COL_NAME not in data_df.columns:
        raise Exception(f"Id column name should be {ID_COL_NAME}")
    
    print(f"Total DataPoints in {input_data_path}: ", len(data_df))
    if "al_iteration_num" not in data_df.columns:
        print("adding al_iteration_num column...")
        data_df["al_iteration_num"] = -1

    return data_df


def annotate_selected_examples(
    input_data_path, uncertain_example_ids, tagged_output_data_path, al_iteration_num=0
):
    """
    uncertain_examples: list of example ids to be annotated
    tagged_output_data_path: path to the tagged output data
    id column name in the input data should be "id"
    """
    data_df = read_and_validate_input_data(input_data_path)

    # filter out the data points which are already tagged
    untagged_data_df = data_df[
        (data_df["al_iteration_num"] == -1)
        & (data_df[ID_COL_NAME].isin(uncertain_example_ids))
    ]
    
    print(f"New DataPoints to annotate: ", len(untagged_data_df))

    selected_data_to_tag = untagged_data_df[["id", INPUT_COL_NAME]].dropna()

    # BATCH_SIZE is added for the inhouse LLMs which might need batching
    for i in tqdm(range(0, len(selected_data_to_tag), BATCH_SIZE)):
        batch_data = selected_data_to_tag[i : i + BATCH_SIZE]
        batch_results = process_batch(batch_data)

        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # write batch results ro json file
        with open(
            f"{TMP_OUTPUT_JSON_DATA_PATH}data_{date_time}.json", mode="w"
        ) as file:
            json.dump(batch_results, file)

        # update the data_df with the iteration number
        data_df.loc[data_df[ID_COL_NAME].isin(batch_data[ID_COL_NAME]),
                    "al_iteration_num"] = al_iteration_num

    print(f"Total DataPoints annotated: ", len(selected_data_to_tag))

    data_df.to_csv(input_data_path, index=False)
    write_consolidated_outputs(tagged_output_data_path)

    return