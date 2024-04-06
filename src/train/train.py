import wandb
import torch
import torch.nn as nn
from torch.utils.data import random_split


from datasets import load_from_disk


from transformers import AutoModel, \
                         TrainingArguments, \
                         Trainer, \
                         AutoModelForTokenClassification, \
                         AutoConfig, \
                         DataCollatorForTokenClassification, \
                         AutoTokenizer

from src.ner_evaluate import compute_all_metrics
from src.utils import get_project_config, read_json_file, get_model_tokenizer
from src.wand_eval_callback import TrainerEvalCallback
from src.models import NERModelWithCRF


project_config = get_project_config()
entity_mapping_path = project_config["data_paths"]["entities_to_label_mapping_path"]
    
entity_mapping = read_json_file(entity_mapping_path)
label2id = entity_mapping["label2id"]

id2label = entity_mapping["id2label"]
id2label = dict((int(k), v) for k, v in id2label.items() if int(k)!=-100)
label2id.pop('<SPECIAL_TOKEN_TAG>')



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


class NERTrainer:
    def __init__(self):
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.project_config = get_project_config()
        self.entity_mapping_path = self.project_config["data_paths"]["entities_to_label_mapping_path"]
        self.entity_mapping = read_json_file(self.entity_mapping_path)
        self.label2id = self.entity_mapping["label2id"]
        self.id2label = self.entity_mapping["id2label"]
        self.id2label = dict((int(k), v) for k, v in self.id2label.items() if int(k)!=-100)
        self.label2id.pop('<SPECIAL_TOKEN_TAG>')
        self.SEED = 42
        self.DATA_SEED = 42
        

    def train_ner_model(self, tagged_input_data_path, previous_model_path, model_output_path, run_name, checkpoint_path=None, use_crf=False, batch_size=8):
        wandb.init(project="GlobalNER", name=run_name)

        def get_model():
            if use_crf:
                model = NERModelWithCRF(pretrained_model_name=previous_model_path, label2id=self.label2id, id2label=self.id2label)
                model = model.to(self.DEVICE)
                return model
            return AutoModelForTokenClassification.from_pretrained(
                previous_model_path,
                id2label=self.id2label,
                label2id=self.label2id
            ).to(self.DEVICE)

        dataset = load_from_disk(tagged_input_data_path)
        
        total_length = len(dataset)
        train_length = int(0.8 * total_length)
        valid_length = total_length - train_length
        train_ds, valid_ds = random_split(dataset, [train_length, valid_length], generator=torch.Generator().manual_seed(self.DATA_SEED))
        dataset_size = len(dataset)
        
        tokenizer = get_model_tokenizer()
        
        
        training_args = self._get_training_arguments(model_output_path, checkpoint_path, batch_size, dataset_size)
        data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")
        trainer = self._get_trainer(training_args, data_collator, train_ds, valid_ds, tokenizer, get_model)
        trainer_callback = TrainerEvalCallback(trainer=trainer, eval_log_sample_size=5, id2label=self.id2label)
        trainer.add_callback(trainer_callback)
        self._train_model(trainer, model_output_path, run_name)

        
    def _get_training_arguments(self,
                            model_output_path,
                            checkpoint_path,
                            batch_size,
                            dataset_size,
                            epoch: int = 15,
                            learning_rate: float = 1e-5,
                            gradient_accumulation_steps: int = 4,
                            eval_accumulation_steps: int = 4,
                            weight_decay: float = 0.01
                           ):

        
        
        total_steps = dataset_size*epoch
        
        per_epoch_eval_count = 4 if epoch<=5 else 1
        count_evals = per_epoch_eval_count*epoch
        eval_steps = int((total_steps/batch_size)/count_evals)
        
        # Return the TrainingArguments object
        return TrainingArguments(
            output_dir = f"{model_output_path}",
            resume_from_checkpoint=checkpoint_path if checkpoint_path else None,
            overwrite_output_dir = True,
            log_level = "error",
            logging_dir='./logs',
            evaluation_strategy = "epoch",
            save_strategy="epoch",
            save_total_limit=2,    
            # push_to_hub = True,
            num_train_epochs = epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=eval_accumulation_steps,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay=weight_decay,
            seed=self.SEED,
            data_seed=self.DATA_SEED,
            load_best_model_at_end=True,
            report_to="wandb"
            )    
        
    def _get_trainer(self, training_args, data_collator, train_ds, valid_ds, tokenizer, get_model):
        # Return the Trainer object
        return Trainer(
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
            model_init=get_model,
            compute_metrics = compute_all_metrics
        )

    def _train_model(self, trainer, model_output_path, run_name):
        wandb.run.name = run_name
        trainer.train()
        trainer.save_model(model_output_path)
        
        
    
