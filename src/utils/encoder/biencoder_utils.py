
import os
import json
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer, AutoModel


class BiEncoderModel(torch.nn.Module):
    def __init__(self, model_name, method='cosine'):
        super(BiEncoderModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method  # Choose between 'cosine' or 'linear'

        if self.method == 'linear':
            hidden_size = self.encoder.config.hidden_size
            # Linear layer for classification
            self.classifier = torch.nn.Linear(hidden_size * 2, 1)  # Binary classification

    def forward(
        self,
        input_ids_a=None,
        attention_mask_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        labels=None,
    ):
        output_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
        output_b = self.encoder(input_ids=input_ids_b, attention_mask=attention_mask_b)

        # Use the [CLS] token representations
        pooled_output_a = output_a.last_hidden_state[:, 0, :]
        pooled_output_b = output_b.last_hidden_state[:, 0, :]

        pooled_output_a = self.dropout(pooled_output_a)
        pooled_output_b = self.dropout(pooled_output_b)

        if self.method == 'cosine':
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(pooled_output_a, pooled_output_b)
            logits = cos_sim.unsqueeze(-1)  # Shape [batch_size, 1]

        elif self.method == 'linear':
            # Concatenate the embeddings and use a linear layer for classification
            combined_output = torch.cat([pooled_output_a, pooled_output_b], dim=1)
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float().unsqueeze(-1))  # Ensure labels match logits shape

        return SequenceClassifierOutput(loss=loss, logits=logits)


# Chroma DB utils

def load_chromaDB_byname(chroma_db_name):
    import chromadb
    # ChromaDB 클라이언트 초기화 및 인코딩된 법률 저장 또는 불러오기
    client = chromadb.PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)

    # 이미 저장된 인코딩이 있으면 불러오기
    chroma_collection = client.get_or_create_collection("quickstart")

    return chroma_collection