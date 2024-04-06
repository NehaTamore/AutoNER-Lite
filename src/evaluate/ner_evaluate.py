import numpy as np
from src.utils import get_project_config, read_json_file
import evaluate
from src.predict import predict_ner_from_hf_dataset
from transformers import AutoModelForTokenClassification
from datasets import load_from_disk
metric = evaluate.load("seqeval")

project_config = get_project_config()
entity_mapping_path = project_config["data_paths"]["entities_to_label_mapping_path"]
    
entity_mapping = read_json_file(entity_mapping_path)
label2id = entity_mapping["label2id"]
id2label = entity_mapping["id2label"]
id2label = dict((int(k), v) for k, v in id2label.items())

def compute_all_metrics(eval_preds):
    
    logits, labels = eval_preds
    
    labels = np.array(labels)
    logits = np.array(logits)
    
    if isinstance(logits[0], int):
        predictions = np.argmax(logits, axis=-1)
    else:
        predictions = logits

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    print("OVERALL F1: ", all_metrics["overall_f1"])
    return {
        "all": all_metrics,
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
    
    
def test_ner_model(test_data_path, model_output_path):
    model = AutoModelForTokenClassification.from_pretrained(model_output_path)
    dataset = load_from_disk(test_data_path)
    tokenised_outputs = predict_ner_from_hf_dataset(dataset, model)
    
    print(compute_all_metrics((tokenised_outputs["prediction_probabilities"], tokenised_outputs["labels"])))
    
    