import torch
from transformers import AutoTokenizer
import pandas as pd
import chromadb

from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm

from src.utils.encoder.utils import MAX_TOKEN_LENGTH
from src.methods.case_augmentation.prompt import generate_case

import time



def find_top_contradictions_with_classifier(article, model, tokenizer, chroma_collection, top_k=10, method="baseline"):
    """
    Function to find top contradictions using classifier method with optional case augmentation.
    
    Args:
    article (str): The input article to check.
    model (torch.nn.Module): Pre-trained model with classifier.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_collection: Chroma DB collection containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """
    if method == "caseaug":
        case = generate_case(None, None, article)
        article = article + " " + case

    # Encode the input article
    inputs = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=MAX_TOKEN_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    with torch.no_grad():
        encoded_article = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoded_article.last_hidden_state[0, 0, :].to(model.encoder.device)  

    # Retrieve encoded laws from Chroma DB
    encoded_laws = chroma_collection.get(include=["embeddings", "documents"])
    law_vectors = np.array(encoded_laws["embeddings"])
    law_documents = encoded_laws["documents"]

    retrieval_start_time = time.time()


    logits_list = []
    for law_vector in tqdm(law_vectors):
        law_vector_tensor = torch.tensor(law_vector, dtype=torch.float32).to(model.encoder.device)
        combined_input = torch.cat([pooled_output, law_vector_tensor], dim=-1).unsqueeze(0)

        with torch.no_grad():
            logits = model.classifier(combined_input)
            logits_list.append(logits.item())




    logits_array = np.array(logits_list)
    top_k_indices = logits_array.argsort()[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]

    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"biencoder 실행 시간: {elapsed_time:.6f}초")


    return pooled_output, top_k_articles


# our original code, bi-encoder.
def find_top_contradictions_with_cosine(article, model, tokenizer, chroma_collection, top_k=10, method="baseline"):
    """
    Function to find top contradictions using cosine similarity method with optional case augmentation.
    
    Args:
    article (str): The input article to check.
    model (torch.nn.Module): Pre-trained model.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_collection: Chroma DB collection containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """
    if method == "caseaug":
        case = generate_case(None, None, article)
        article = article + " " + case

    # Encode the input article
    inputs = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=MAX_TOKEN_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    with torch.no_grad():
        encoded_article = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoded_article.last_hidden_state[0, 0, :].cpu().numpy()

    # Retrieve encoded laws from Chroma DB
    encoded_laws = chroma_collection.get(include=["embeddings", "documents"])
    law_vectors = np.array(encoded_laws["embeddings"])
    law_documents = encoded_laws["documents"]

    retrieval_start_time = time.time()


    similarities = cosine_similarity(
        torch.tensor(pooled_output).unsqueeze(0),
        torch.tensor(law_vectors)
    ).numpy().flatten()

    top_k_indices = similarities.argsort()[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]

    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"biencoder 실행 시간: {elapsed_time:.6f}초")

    return pooled_output, top_k_articles


def binary_retriever(model_path, laws_csv_path, chroma_db_name, article_to_check, classification_method="cosine", top_k=500, case_augmentation_method = "baseline", batch_size = 8):
    """
    Bi-encoder retriever function to find contradictions in legal texts.

    Args:
    model_path (str): Path to the trained model directory.
    laws_csv_path (str): Path to the laws.csv file.
    chroma_db_name (str): Name of the Chroma DB where encodings will be stored.
    article_to_check (str): The article to check for contradictions.
    classification_method (str): Method to use for classification ('cosine' or 'classification_model').
    top_k_val (int): Number of top-k articles to retrieve.

    Returns:
    List of top-k articles that contradict the input article.
    """

    model = torch.load(model_path + "/model.pth")
    model.eval()  # 모델을 평가 모드로 설정
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    laws_csv = laws_csv_path
    laws_df = pd.read_csv(laws_csv)

    from src.utils.encoder.biencoder_utils import load_chromaDB_byname
    chroma_collection = load_chromaDB_byname(chroma_db_name)

    # 인코딩이 없으면 새로 인코딩하여 저장
    if chroma_collection.count() == 0:
        
        ids = []
        documents = []
        embeddings = []

        idx = 0
        all_articles = []

        for _, row in tqdm(laws_df.iterrows()):
            article = row["contents"]
            if case_augmentation_method == "caseaug":
                case = generate_case(None, None, article)
                article = article + "\n[CASE]\n" + case    
            all_articles.append(article)
            ids.append(str(idx))
            idx += 1

        # Process in batches
        for batch_start in tqdm(range(0, len(all_articles), batch_size)):
            batch_articles = all_articles[batch_start:batch_start + batch_size]
            batch_inputs = tokenizer(batch_articles, add_special_tokens=True, max_length=MAX_TOKEN_LENGTH, 
                                     padding="max_length", truncation=True, return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(model.encoder.device)
            attention_mask = batch_inputs["attention_mask"].to(model.encoder.device)

            with torch.no_grad():
                encoded_articles = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled_outputs = encoded_articles.last_hidden_state[:, 0, :].cpu().numpy()

            # Collect embeddings and documents
            embeddings.extend(pooled_outputs)
            documents.extend(batch_articles)

        chunk_size = 1024
        total_chunks = len(embeddings) // chunk_size + 1
        embeddings = [e.flatten().tolist() for e in embeddings]

        for chunk_idx in tqdm(range(total_chunks)):
            start_idx = chunk_idx * chunk_size
            end_idx = (chunk_idx + 1) * chunk_size

            chunk_embeddings = embeddings[start_idx:end_idx]
            chunk_ids = ids[start_idx:end_idx]
            chunk_docs = documents[start_idx:end_idx]

            chroma_collection.add(
                documents=chunk_docs,
                embeddings=chunk_embeddings,
                ids=chunk_ids,
            )
        


    # 특정 article을 입력하여 모순된 법률 찾기
    if classification_method == "cosine":
        article_vector, top_contradictions = find_top_contradictions_with_cosine(article_to_check, model, tokenizer, chroma_collection, top_k=top_k, method=case_augmentation_method)
    else:
        article_vector, top_contradictions = find_top_contradictions_with_classifier(article_to_check, model, tokenizer, chroma_collection, top_k=top_k, method=case_augmentation_method)


    if case_augmentation_method == "caseaug":
        top_contradictions=[t.split("[CASE]")[0] for t in top_contradictions]

    

    return article_vector, top_contradictions


