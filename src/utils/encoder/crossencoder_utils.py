

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModel

from src.utils.utils import article_key_function
from src.utils.encoder.utils import ThreeLayerClassifier


class CrossEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(CrossEncoderModel, self).__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except:
            # Model type should be one of ... error
            from transformers import AutoModelForCausalLM
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id

        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = self.encoder.config.hidden_size
        # Linear layer for classification
        # self.classifier = ThreeLayerClassifier(input_size=hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, 1)  # Binary classification

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        # Input will now be concatenated pairs of sequences (cross-encoder)
        # Example: [CLS] seqA tokens ... [SEP] seqB tokens ...
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        
        pooled_output = self.dropout(pooled_output)


        logits = self.classifier(pooled_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))  # Ensure labels match logits shape

        return SequenceClassifierOutput(loss=loss, logits=logits)




class NLIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, method="baseline"):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.method = method

        # bi-encoder 에서 가져옴
        self.label_map = {True: 1, False: 0}

    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):

        if self.method =="case-augmentation":
            from src.methods.case_augmentation.prompt import generate_case
            premise = self.dataframe.iloc[index]["article1"] + "\ncase:\n"+ generate_case(None, None, self.dataframe.iloc[index]["article1"])
            hypothesis = self.dataframe.iloc[index]["article2"]+"\ncase:\n"+ generate_case(None, None, self.dataframe.iloc[index]["article2"], case_idx=self.dataframe.iloc[index]['case_idx'])
        
        elif self.method == "case-concat-augmentation":
            from src.methods.case_augmentation.prompt import generate_case
            article_key = article_key_function(self.dataframe.iloc[index]["article1"])+"-"+article_key_function(self.dataframe.iloc[index]["article2"])
            premise = self.dataframe.iloc[index]["article1"]
            hypothesis = "case:\n"+ generate_case(None,None, article=self.dataframe.iloc[index]["article1"], article_key=article_key) +"\n" + self.dataframe.iloc[index]["article2"]

        # baseline   
        else:
            premise = self.dataframe.iloc[index]["article1"]
            hypothesis = self.dataframe.iloc[index]["article2"]
        label = self.dataframe.iloc[index]["answer"]
        label = 1 if label else 0  # True -> 1, False -> 0
        
        encoding = self.tokenizer.encode_plus(
            text=premise,
            text_pair=hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def print_label_counts(self):
        true_count = (self.dataframe["answer"] == True).sum()
        false_count = (self.dataframe["answer"] == False).sum()
        print(f"True labels: {true_count}")
        print(f"False labels: {false_count}")
