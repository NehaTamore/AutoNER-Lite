import torch
import torch.nn as nn

from transformers import AutoModel, \
                         TrainingArguments, \
                         Trainer, \
                         AutoModelForTokenClassification, \
                         AutoConfig, \
                         DataCollatorForTokenClassification, \
                         AutoTokenizer

from allennlp_light.modules import ConditionalRandomField
from allennlp_light.modules.conditional_random_field.conditional_random_field import allowed_transitions



class NERModelWithCRF(nn.Module):
    def __init__(self, pretrained_model_name, label2id, id2label):
        super(NERModelWithCRF, self).__init__()
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(label2id)

        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.embedding_dim = AutoConfig.from_pretrained(pretrained_model_name).hidden_size
        self.classifier = nn.Linear(self.embedding_dim, self.num_labels)
        
        self.crf = ConditionalRandomField(
            self.num_labels,
            include_start_end_transitions=False
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
            
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        
        mask = self.create_combined_mask(labels, attention_mask)
        predictions = self.viterbi_decode(logits, mask)
        predictions_tensor = self.pad_predictions(predictions, labels.shape[1])
        
        if labels is not None:
            self.validate_inputs(labels, attention_mask)
            adjusted_labels = self.adjust_labels_for_crf(labels)
            log_likelihood = self.crf(logits, adjusted_labels, mask=mask)

            return {"loss": -log_likelihood, "predictions": predictions_tensor}
        else:
            return {"predictions": predictions_tensor}

    def validate_inputs(self, labels, attention_mask):
        if not isinstance(labels, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise ValueError("Labels and attention mask must be torch.Tensors")
    
    def create_combined_mask(self, labels, attention_mask):
        labels_mask = (labels != -100).type(torch.bool)
        return attention_mask & labels_mask
    
    def adjust_labels_for_crf(self, labels):
        adjusted_labels = labels.clone()
        adjusted_labels[labels == -100] = 0   # this label shall be ignored, but used as placeholder
        return adjusted_labels

    def viterbi_decode(self, logits, mask):
        return [x[0] for x in self.crf.viterbi_tags(logits, mask)]

    def pad_predictions(self, predictions, max_length=None):
        if max_length is None:
            max_length = max(len(seq) for seq in predictions)
            
        padded_predictions = [[-100] + seq + [-100]*(max_length - len(seq) - 1) for seq in predictions]
        return torch.tensor(padded_predictions, dtype=torch.long)

