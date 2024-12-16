# /lawDB-LLM 아래에서 시작






# KoBigBird, batch 8
python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode test --tag kbb-baseline

python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode test --tag kbb-caseaug --method case-augmentation


# Qwen 0.5B, batch 2
python ./src/encoders/bi_encoder/train/finetune.py --mode test --tag qwen2-0.5b-baseline  --model Qwen/Qwen2-0.5B-Instruct

python ./src/encoders/bi_encoder/train/finetune.py --mode test --tag qwen2-0.5b-caseaug  --method case-augmentation --model Qwen/Qwen2-0.5B-Instruct



# Qwen 1.5B batch 2
python ./src/encoders/bi_encoder/train/finetune.py --mode train --tag qwen2-1.5b-baseline  --model Qwen/Qwen2-1.5B-Instruct

python ./src/encoders/bi_encoder/train/finetune.py --mode train --tag qwen2-1.5b-caseaug --method case-augmentation --model Qwen/Qwen2-1.5B-Instruct


# Phi3 3.8B
python ./src/encoders/bi_encoder/train/finetune.py --mode train --tag phi3mini-baseline  --model microsoft/Phi-3-mini-4k-instruct

python ./src/encoders/bi_encoder/train/finetune.py --mode train --tag phi3mini-caseaug --method case-augmentation --model microsoft/Phi-3-mini-4k-instruct



# biencoder train

# python ./src/encoders/bi_encoder/train/finetune.py --model monologg/kobigbird-bert-base --mode train --tag kbb-caseaug --method case-augmentation