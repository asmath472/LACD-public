# KBB


echo kbb-baseline "naive step 1 and 3"
python ./src/encoders/cross_encoder/train/finetune.py --mode train --tag kbb-baseline --method baseline --epoch 3


echo kbb-baseline-gat "naive step 1 and CAM-Re2 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-gat --gnn_method gat --chroma_db_name kbb-baseline --case_augmentation_method baseline --epoch 3


echo kbb-baseline-gat-caseaugembds "CAM-Re2 step 1 and 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag kbb-baseline-gat-caseaugembds --gnn_method gat --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3

# QWEN


echo qwen-baseline "naive step 1 and 3"
python ./src/encoders/cross_encoder/train/finetune.py --mode train --tag qwen2-0.5-baseline --method baseline --epoch 3 --model Qwen/Qwen2-0.5B


echo qwen2-0.5-baseline-gat "naive step 1 and CAM-Re2 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag qwen2-0.5-baseline-gat --gnn_method gat --chroma_db_name kbb-baseline --case_augmentation_method baseline --epoch 3 --model Qwen/Qwen2-0.5B

echo qwen2-0.5-baseline-gat-caseaugembds "CAM-Re2 step 1 and 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag qwen2-0.5-baseline-gat-caseaugembds --gnn_method gat --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3 --model Qwen/Qwen2-0.5B




# LLAMA
echo llama3-1b-baseline "naive step 1 and 3"
python ./src/encoders/cross_encoder/train/finetune.py --mode train --tag llama3-1b-Instruct-baseline --method baseline --epoch 3 --model meta-llama/Llama-3.2-1B



echo llama3-1b-baseline-gat "naive step 1 and CAM-Re2 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag llama3-1b-baseline-gat --gnn_method gat --chroma_db_name kbb-baseline --case_augmentation_method baseline --epoch 3 --model meta-llama/Llama-3.2-1B


echo llama3-1b-baseline-gat-caseaugembds "CAM-Re2 step 1 and 3"
python ./src/methods/LawGNN/train/crossencoder_finetune.py --tag llama3-1b-baseline-gat-caseaugembds --gnn_method gat --chroma_db_name kbb-caseaug --case_augmentation_method baseline --epoch 3 --model meta-llama/Llama-3.2-1B


