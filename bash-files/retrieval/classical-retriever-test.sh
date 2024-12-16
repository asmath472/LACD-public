

# TF-IDF
python ./src/main.py --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method tfidf

# BM25
python ./src/main.py --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bm25

# TF-IDF benchmark result
python ./src/main.py --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method tfidf --mode test-benchmark

# BM25 benchmark result 