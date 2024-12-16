python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bi-only

python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug --chroma_db_name kbb-caseaug --biencoder_method caseaug --retrieval_method bi-only
