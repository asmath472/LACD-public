import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm, trange
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.methods.case_augmentation.prompt import generate_case
import numpy as np

from src.utils.encoder.utils import MAX_TOKEN_LENGTH
from src.utils.utils import article_key_function

import time

def cross_retriever(article, top_k_articles, model_path, article_network:ArticleNetwork, top_k=10, batch_size=64, method = "baseline", index_method = "none", crossencoder_index_db="kbb-baseline", article_vector = None):
    """
    Function to use a cross-encoder to distinguish contradictions in top-k retrieved articles using batch processing.

    Major args:
    article (str): The original article to compare against top-k retrieved articles.
    top_k_articles (list): List of top-k retrieved articles.
    model_path (str): Path to the cross-encoder model.
    top_k (int): The number of top articles to return based on contradiction score.
    batch_size (int): The number of articles to process in one batch.

    Returns:
    List of (article, score) where score represents the classification score of contradiction.
    """
    # Load tokenizer and cross-encoder model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cross_encoder_model = torch.load(model_path + "/model.pth")
    cross_encoder_model.eval()

    if article_key_function(article) not in article_network.article_key_to_idx.keys():
        article_idx = article_network.add_article_node(article)
    else:
        article_idx = article_network.article_key_to_idx[article_key_function(article)]

    if index_method != "none":
        if isinstance(article_vector, np.ndarray):
            article_vector = torch.tensor(article_vector).to(cross_encoder_model.vector_tensor.device)
        if article_vector.dim() == 1: # type: ignore
            article_vector = article_vector.unsqueeze(0)  # type: ignore # (1, D) 형태로 변환
        cross_encoder_model.vector_tensor = torch.cat([cross_encoder_model.vector_tensor, article_vector], dim=0) # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_encoder_model.to(device)

    retrieval_start_time = time.time()


    if method == "caseaug":
        case = generate_case(None, None, article)
        article = article+"\ncase:\n"+case

    # List to store results
    contradictions = []

    # Process top_k_articles in batches
    for i in range(0, len(top_k_articles), batch_size):
        batch_articles = top_k_articles[i:i+batch_size]

        if method == "caseaug":
            original_articles = batch_articles
            batch_articles = [b+"\ncase:\n"+generate_case(None, None, b) for b in batch_articles]

        # Tokenize the batch of (article, batch_articles) pairs
        inputs = tokenizer.batch_encode_plus(
            [(article, retrieved_article) for retrieved_article in batch_articles],
            add_special_tokens=True,
            max_length=MAX_TOKEN_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move input tensors to the appropriate device
        input_ids = inputs['input_ids'].to(device) # type: ignore
        attention_mask = inputs['attention_mask'].to(device) # type: ignore


        article2_idx_list = [
            article_network.article_key_to_idx[article_key_function(retrieved_article)]
            for retrieved_article in batch_articles
        ]
        article1_idx_tensor = torch.tensor([article_idx for retrieved_article in batch_articles]).to(device)
        article2_idx_tensor = torch.tensor(article2_idx_list).to(device)  # Convert to tensor and move to device


        # Get model prediction for the batch
        with torch.no_grad():
            if index_method != "none":
                outputs = cross_encoder_model(article1_idx=article1_idx_tensor,article2_idx=article2_idx_tensor,
                input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = cross_encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if logits.shape[1] == 1:
                # 한 개의 확률을 출력하는 경우 (예: [batch_size, 1])
                probs = torch.sigmoid(logits).cpu().numpy()  # Sigmoid로 변환하여 확률 얻기
            else:
                # 두 개의 확률을 출력하는 경우 (예: [batch_size, 2])
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # Softmax로 변환하여 확률 얻기

        # Process the results for each article in the batch
        for idx, retrieved_article in enumerate(batch_articles):
            if logits.shape[1] == 1:
                # Sigmoid로 변환된 단일 값 (True일 확률)
                contradiction_score = probs[idx][0]
            else:
                # Softmax로 변환된 두 번째 값 (True일 확률)
                contradiction_score = probs[idx][1]  # 두 번째 값이 True일 확률
            if method == "caseaug":
                contradictions.append((original_articles[idx], contradiction_score))
                
            else:
                contradictions.append((retrieved_article, contradiction_score))

    # Sort contradictions based on the contradiction score in descending order
    contradictions = sorted(contradictions, key=lambda x: x[1], reverse=True)
    
    contradictions = [c for c in contradictions if c[1] >= 0.5]
    
    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"crossencoder 실행 시간: {elapsed_time:.6f}초")


    # Return the top_k articles with the highest contradiction scores
    return [c[0] for c in contradictions[:top_k]]


def noLM_cross_retriever(article, top_k_articles, model_path, article_network:ArticleNetwork, top_k=10, batch_size=64, method = "baseline", index_method = "none", crossencoder_index_db="kbb-baseline", article_vector = None):
    """
    Function to use a cross-encoder to distinguish contradictions in top-k retrieved articles using batch processing.

    Major args:
    article (str): The original article to compare against top-k retrieved articles.
    top_k_articles (list): List of top-k retrieved articles.
    model_path (str): Path to the cross-encoder model.
    top_k (int): The number of top articles to return based on contradiction score.
    batch_size (int): The number of articles to process in one batch.

    Returns:
    List of (article, score) where score represents the classification score of contradiction.
    """
    # Load tokenizer and cross-encoder model

    cross_encoder_model = torch.load(model_path + "/model.pth")
    cross_encoder_model.eval()

    if article_key_function(article) not in article_network.article_key_to_idx.keys():
        article_idx = article_network.add_article_node(article)
        if index_method != "none":
            if isinstance(article_vector, np.ndarray):
                article_vector = torch.tensor(article_vector).to(cross_encoder_model.vector_tensor.device)
            if article_vector.dim() == 1: # type: ignore
                article_vector = article_vector.unsqueeze(0)  # type: ignore # (1, D) 형태로 변환
            cross_encoder_model.vector_tensor = torch.cat([cross_encoder_model.vector_tensor, article_vector], dim=0) # type: ignore
    else:
        article_idx = article_network.article_key_to_idx[article_key_function(article)]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_encoder_model.to(device)

    retrieval_start_time = time.time()

    if method == "caseaug":
        case = generate_case(None, None, article)
        article = article+"\ncase:\n"+case

    # List to store results
    contradictions = []

    # Process top_k_articles in batches
    for i in range(0, len(top_k_articles), batch_size):
        batch_articles = top_k_articles[i:i+batch_size]

        if method == "caseaug":
            original_articles = batch_articles
            batch_articles = [b+"\ncase:\n"+generate_case(None, None, b) for b in batch_articles]

        # Tokenize the batch of (article, batch_articles) pairs
        article2_idx_list = [
            article_network.article_key_to_idx[article_key_function(retrieved_article)]
            for retrieved_article in batch_articles
        ]
        article1_idx_tensor = torch.tensor([article_idx for retrieved_article in batch_articles]).to(device)
        article2_idx_tensor = torch.tensor(article2_idx_list).to(device)  # Convert to tensor and move to device


        # Get model prediction for the batch
        with torch.no_grad():
            if index_method != "none":
                outputs = cross_encoder_model(article1_idx=article1_idx_tensor,article2_idx=article2_idx_tensor)
            logits = outputs.logits

            if logits.shape[1] == 1:
                # 한 개의 확률을 출력하는 경우 (예: [batch_size, 1])
                probs = torch.sigmoid(logits).cpu().numpy()  # Sigmoid로 변환하여 확률 얻기
            else:
                # 두 개의 확률을 출력하는 경우 (예: [batch_size, 2])
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # Softmax로 변환하여 확률 얻기

        # Process the results for each article in the batch
        for idx, retrieved_article in enumerate(batch_articles):
            if logits.shape[1] == 1:
                # Sigmoid로 변환된 단일 값 (True일 확률)
                contradiction_score = probs[idx][0]
            else:
                # Softmax로 변환된 두 번째 값 (True일 확률)
                contradiction_score = probs[idx][1]  # 두 번째 값이 True일 확률
            if method == "caseaug":
                contradictions.append((original_articles[idx], contradiction_score))
                
            else:
                contradictions.append((retrieved_article, contradiction_score))

    # Sort contradictions based on the contradiction score in descending order
    contradictions = sorted(contradictions, key=lambda x: x[1], reverse=True)
    
    contradictions = [c for c in contradictions if c[1] >= 0.5]
    
    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"crossencoder 실행 시간: {elapsed_time:.6f}초")

    # Return the top_k articles with the highest contradiction scores
    return [c[0] for c in contradictions[:top_k]]