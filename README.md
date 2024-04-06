 # AutoNER-Lite
AutoNER-Lite is an end to end, lightweight Named Entity Recognition (NER) system designed to efficiently train domain-specific entities with clear distinction and accuracy without human annotation. The system is optimized for minimal data requirements and reduced latency during training and inference.

AutoNER-Lite addresses three critical areas within the NER landscape to enhance the accuracy, scalability, and data efficiency of entity recognition:

#### Overlap and Subjectivity in Entity Tagging
- **Objective Annotations**: By implementing clear guidelines for entity definitions, AutoNER-Lite reduces overlap and subjectivity, achieving more consistent and objective annotations across various domains.

#### Minimal Data Requirements
- **Data Efficiency**: AutoNER-Lite is optimized for minimal data requirements, employing techniques like automated annotation correction and domain overlap quantification to train effectively on limited datasets, thus lowering the barrier for NER projects across any domain.  

#### Scalability
- **Adaptive Scaling**: AutoNER-Lite takes a distinct approach by utilizing smaller pretrained models. By feeding these models sufficient and precisely selected training data, AutoNER-Lite ensures that the model remains lightweight and scalable, particularly addressing the bottleneck of inference time.

## Installation Guide

### Steps:

1. Install Dependencies:
   ```
   pip install [dependencies]
   ```

2. Configure project_config.json:
   ```
   {
       "configuration_key": "configuration_value"
   }
   ```

3. Run the NER Training Pipeline via main.ipynb:
   

## Usage Instructions

- Define Entities: It is crucial to define domain-specific entities (e.g., comorbidity, existing_condition, disease, pre-existing condition) with clear and objective annotation for efficient and accurate NER.
- Proportion of Data Annotation: Choose the proportion of untagged data to be annotated in each iteration based on data availability and project needs.
- CRF Layer Latency: Note that the CRF layer may add latency during both training and inference.

## Configuration

The project configuration is defined in the `project_config.json` file.

## Project Structure

```
.
├── src/
|   ├── dataset/
|   |   ├── al_data_selection.py
|   |   ├── hf_dataset_creation.py
|   |   ├── data_annotation.py
|   |   ├── prompts.py
|   |   └── annotation_correction.py
|   |
|   ├── train/
|   |   ├── models.py
|   |   ├── train.py
|   |   └── wand_eval_callback.py
|   |
|   └── evaluate/
|       ├── ner_evaluate.py
|       └── predict.py
|
├── main.ipynb
├── project_config.json
├── artefacts/
├── wandb/
└── data/

```

## Contributing Guidelines

We encourage collaboration and contribution to AutoNER-Lite. To contribute, please follow the outlined process for submitting issues, feature requests, and pull requests. Additionally, adhere to the coding standards, testing procedures, and other relevant guidelines.


## Developments

AutoNER-Lite is continuously evolving in the NLP model training field. The potential future developments include:

- Further optimization of the training process to reduce data requirements
- Integration of WANDB Sweeps for automated hyperparameter tuning
- Incorporation of LLM for entity definitions
- Implementation of advanced active learning strategies
- Support for selecting the initial LM by quantifying domain overlap
- Integration of Cleanlab for LLM annotation correction


## Credits

AutoNER-Lite leverages the capabilities of allennlp-light and huggingface.

## Contact Information

For any inquiries or assistance, please contact Neha Tamore at nehatamore87@gmail.com.

## Example
The following is the output of NER system trained on medical case reports, without any postprocessing of token-labels.
![Medical NER](./medical_ner.png "NER")
