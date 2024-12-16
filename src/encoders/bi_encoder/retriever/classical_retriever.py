from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi
from src.methods.case_augmentation.prompt import generate_case
from src.utils.encoder.biencoder_utils import load_chromaDB_byname

def find_top_contradictions_with_tfidf(article, chroma_db_name, top_k=10, method="baseline"):
    """
    Function to find top contradictions using TF-IDF retriever.
    
    Args:
    article (str): The input article to check.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_db_name: Chroma DB name containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """

    chroma_collection = load_chromaDB_byname(chroma_db_name)


    if method == "caseaug":
        case = generate_case(None, None, article)
        article = article + " " + case

    # Retrieve encoded laws from Chroma DB
    encoded_laws = chroma_collection.get(include=["embeddings", "documents"])
    law_documents = encoded_laws["documents"]

    # TF-IDF Vectorization
    corpus = law_documents + [article]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    article_vector = tfidf_matrix[-1]  # Last vector corresponds to the input article
    law_vectors = tfidf_matrix[:-1]   # All but the last are law documents

    similarities = np.dot(law_vectors, article_vector.T).toarray().flatten()
    top_k_indices = similarities.argsort()[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]

    return top_k_articles


def find_top_contradictions_with_bm25(article, chroma_db_name, top_k=10, method="baseline"):
    """
    Function to find top contradictions using BM25 retriever.
    
    Args:
    article (str): The input article to check.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_db_name: Chroma DB name containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """


    chroma_collection = load_chromaDB_byname(chroma_db_name)

    if method == "caseaug":
        case = generate_case(None, None, article)
        article = article + " " + case

    # Retrieve encoded laws from Chroma DB
    encoded_laws = chroma_collection.get(include=["embeddings", "documents"])
    law_documents = encoded_laws["documents"]

    # Tokenize and apply BM25
    corpus = [doc.split() for doc in law_documents]  # Split documents into words
    bm25 = BM25Okapi(corpus)

    # Tokenize the input article
    article_tokens = article.split()

    # Get BM25 scores
    scores = bm25.get_scores(article_tokens)
    
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    top_k_articles = [law_documents[i] for i in top_k_indices]

    return top_k_articles
