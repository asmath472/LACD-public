

# for biencoder_method in "baseline"

# do


# echo bi-${biencoder_method} no-index
# python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-${biencoder_method}  --chroma_db_name kbb-${biencoder_method} --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method ${biencoder_method} --mode test-benchmark

# echo bi-${biencoder_method} gat index
# python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/qwen2-0.5-baseline-gat --biencoder_model_path ./data/models/LACD-bi/kbb-${biencoder_method}  --chroma_db_name kbb-${biencoder_method} --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method ${biencoder_method} --mode test-benchmark

# done

# for biencoder_method in "caseaug" 

# do


# echo bi-${biencoder_method} no-index
# python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-${biencoder_method}  --chroma_db_name kbb-${biencoder_method} --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method ${biencoder_method} --mode test-benchmark

# echo bi-${biencoder_method} gat index
# python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/qwen2-0.5-baseline-gat-${biencoder_method}embds --biencoder_model_path ./data/models/LACD-bi/kbb-${biencoder_method}  --chroma_db_name kbb-${biencoder_method} --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method ${biencoder_method} --mode test-benchmark

# done


# real-case mix
echo bi-caseaug no-index
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug  --chroma_db_name kbb-caseaug-realcasemix --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method caseaug --mode test-benchmark

echo bi-caseaug gat index
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/qwen2-0.5-baseline-gat-caseaugembds --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug  --chroma_db_name kbb-caseaug --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method caseaug --mode test-benchmark




# python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/qwen2-0.5-baseline-gat-caseaugembds --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug  --chroma_db_name kbb-caseaug --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method baseline --biencoder_method caseaug

# python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/qwen2-0.5-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline  --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline
