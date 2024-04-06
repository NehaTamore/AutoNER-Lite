from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np
import pandas as pd
from seqeval.metrics import f1_score, classification_report, accuracy_score
import evaluate
import wandb

metric = evaluate.load("seqeval")

class TrainerEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_log_sample_size, id2label):
        self.eval_dataset = trainer.eval_dataset
        self.tokenizer = trainer.tokenizer
        self.trainer = trainer
        self.eval_log_sample_size = eval_log_sample_size
        self.sample_indices = np.random.choice(len(self.eval_dataset), self.eval_log_sample_size, replace=False)
        self.eval_pred_df = pd.DataFrame({"ids": self.sample_indices})
        self.step = 0
        self.id2label = id2label
        
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = self.trainer
        
        predictions, true_labels, _ = trainer.predict(self.eval_dataset)
        
        if not hasattr(trainer.model, 'crf'):
            predictions = np.argmax(predictions, axis=-1)
        
        
        
        predictions_tags = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, true_labels)
        ]
        
        true_tags = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, true_labels)
        ]
        
        token_prediction_logs = []
        for index in self.sample_indices:
            input_tokens_ids = self.eval_dataset[index]['input_ids']
            true_sample_labels = self.eval_dataset[index]['labels']
            sample_predictions = predictions[index]
            
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_tokens_ids)
            temp = ""
            for token, label, pred in zip(input_tokens, true_sample_labels, sample_predictions):
                if label!=-100 and pred!=-100:
                    temp += str((token, self.id2label[label], self.id2label[pred]))
                
            token_prediction_logs.append(temp)
            
        accuracy = accuracy_score(true_tags, predictions_tags)
        all_metrics = metric.compute(predictions=predictions_tags, references=true_tags)
        
        self.eval_pred_df[f"predictions_{self.step}"] = token_prediction_logs
        
        
        table = wandb.Table(dataframe=self.eval_pred_df)
        
        wandb.log({f"predictions": table})

        accuracy = accuracy_score(true_tags, predictions_tags)
        all_metrics = metric.compute(predictions=predictions_tags, references=true_tags)

        self.step += 1
        wandb.log({"F1": all_metrics, "ACCURACY": accuracy})
        
        