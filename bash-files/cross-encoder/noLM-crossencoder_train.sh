
for method in "gat" "vanilla"

do
echo kbb-caseaug-${method}-noLM
python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag kbb-caseaug-${method}-noLM --gnn_method ${method} --chroma_db_name kbb-caseaug --epoch 10

echo kbb-baseline-${method}-noLM
python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag kbb-baseline-${method}-noLM --gnn_method ${method} --chroma_db_name kbb-baseline --epoch 10

done


python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag kbb-realcasemix-gat-noLM --gnn_method gat --chroma_db_name kbb-caseaug-realcasemix --gnn_method vanilla --epoch 10 

python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag kbb-realcasemix-gat-noLM --gnn_method gat --chroma_db_name kbb-caseaug-realcasemix --gnn_method gat --epoch 10 


# python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag openai-3small-gat-noLM --gnn_method gat --chroma_db_name openai-3small --epoch 10 
