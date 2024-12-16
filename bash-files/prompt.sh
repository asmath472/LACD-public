
# GPT-3.5-turbo
# python ./src/main.py --answer_save True --method case-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-augmented/small-prompt/gpt-3.5-turbo-casegenaug.jsonl
# python ./src/main.py --answer_save True --method baseline --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-augmented/small-prompt/gpt-3.5-turbo-baseline.jsonl

# GPT-4o-mini
# python ./src/main.py --model gpt-4o-mini --answer_save True --method case-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-augmented/small-prompt/gpt-4o-mini-casegenaug.jsonl
# python ./src/main.py --model gpt-4o-mini --answer_save True --method baseline --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-augmented/small-prompt/gpt-4o-mini-baseline.jsonl

# llama 3.1 8B Korean
# python ./src/main.py --model beomi/Llama-3-Open-Ko-8B --answer_save True --method case-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/llama31-8B-Ko-casegenaug.jsonl
# python ./src/main.py --model beomi/Llama-3-Open-Ko-8B --answer_save True --method baseline --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/llama31-8B-Ko-baseline.jsonl

# Qwen2-1.5B
# python ./src/main.py --model Qwen/Qwen2-1.5B-Instruct --answer_save True --method case-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/Qwen2-1.5B-casegenaug.jsonl
# python ./src/main.py --model Qwen/Qwen2-1.5B-Instruct --answer_save True --method baseline --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/Qwen2-1.5B-baseline.jsonl

# Qwen2-8B
# python ./src/main.py --model Qwen/Qwen2-7B-Instruct --answer_save True --method case-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/Qwen2-7B-casegenaug.jsonl
# python ./src/main.py --model Qwen/Qwen2-7B-Instruct --answer_save True --method baseline --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/Qwen2-7B-baseline.jsonl

# Qwen2-72B
python ./src/main.py --model Qwen/Qwen2-72B-Instruct --answer_save True --method case-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/Qwen2-72B-casegenaug.jsonl
python ./src/main.py --model Qwen/Qwen2-72B-Instruct --answer_save True --method baseline --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/Qwen2-72B-baseline.jsonl


# case aug concatenate
# python ./src/main.py --answer_save True --method case-concat-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/gpt-3.5-turbo-casecongenaug.jsonl
# python ./src/main.py --model gpt-4o-mini --answer_save True --method case-concat-gen-aug --dataset SMALL_DATASET --save_path ./outputs/LACD-prompt/small-prompt/gpt-4o-mini-casecongenaug.jsonl
