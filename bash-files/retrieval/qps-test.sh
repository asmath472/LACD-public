
echo baseline test!

for top_k in 1 2 5 10 25 50 100 200 300 400
do
echo "top-k: ${top_k}"
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/kbb-baseline --biencoder_model_path ./data/models/LACD-bi/kbb-baseline  --chroma_db_name kbb-baseline --retrieval_method hybrid --crossencoder_index_method none --crossencoder_method baseline --biencoder_method baseline --biencoder_top_k ${top_k}

done


echo ours test!

for top_k in 1 2 5 10 25 50 100 200 300 400
do
echo "top-k: ${top_k}"
python ./src/main.py --crossencoder_model_path ./data/models/LACD-cross/gnns/kbb-baseline-gat-caseaugembds --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug  --chroma_db_name kbb-caseaug --retrieval_method hybrid --crossencoder_index_method gat --crossencoder_method caseaug --biencoder_method caseaug --biencoder_top_k ${top_k}

done
