
# GraphSAGE
echo kbb-baseline-graphsage-caseaugembds "CAM-Re2 step 1 and 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-graphsage-caseaugembds --gnn_method graphsage --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3


echo kbb-baseline-graphsage "naive step 1 and CAM-Re2 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-graphsage --gnn_method graphsage --chroma_db_name kbb-baseline --case_augmentation_method baseline --epoch 3


# GCN
echo kbb-baseline-gcn-caseaugembds "CAM-Re2 step 1 and 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-gcn-caseaugembds --gnn_method gcn --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3


echo kbb-baseline-gcn "naive step 1 and CAM-Re2 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-gcn --gnn_method gcn --chroma_db_name kbb-baseline --case_augmentation_method baseline --epoch 3