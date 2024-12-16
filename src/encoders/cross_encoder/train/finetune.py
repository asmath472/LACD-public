# contradiction detection model 을 훈련하기 위한 코드
# 여기에서는 Full-fine-tune 만을 다룬다.

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter # type: ignore
from src.utils.encoder.crossencoder_utils import CrossEncoderModel, NLIDataset
from src.utils.encoder.utils import data_augmentation, balance_dataframe, MAX_TOKEN_LENGTH, compute_metrics, TensorBoardCallback
from src.utils.utils import seed_everything, SEED


if __name__ == "__main__":
    seed_everything(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="monologg/kobigbird-bert-base")
    parser.add_argument("--mode", type=str, help="train, test, or inference", default="train")
    
    # False, trainCivil, trainCriminal
    parser.add_argument("--model_save_path", type=str, help="a path for saving model. do not use None as the name", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")
    parser.add_argument("--data_augmentation", type=str, help="None, augmentation, noA, noB, noswap, or nobalance", default="None")

    parser.add_argument("--case_multiplier", type = int, default=1)
    parser.add_argument("--epoch", type = int, default=3)

    parser.add_argument("--method", type=str, help="baseline or case-augmentation or case-concat-augmentation", default="baseline")

    args = parser.parse_args()

    case_multiplier = args.case_multiplier
    model_name = args.model
    tag = args.tag
    epoch_num = args.epoch
    
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.config.pad_token_id = model.config.eos_token_id

    model = CrossEncoderModel(model_name=model_name)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    tokenizer = model.tokenizer
    model.encoder.resize_token_embeddings(len(tokenizer))

    # tokenizer.padding_side = "right"

    if args.method == "case-augmentation" or args.method == "case-concat-augmentation":
        from src.methods.case_augmentation.prompt import case_cache_start, case_cache_end
        case_cache_start()


    # 분할된 인덱스를 사용하여 train, test 데이터프레임 생성
    train_df = pd.read_json('./data/datasets/LACD-biclassification/train-test-divide/train.jsonl', lines=True)
    test_df = pd.read_json('./data/datasets/LACD-biclassification/train-test-divide/test.jsonl', lines=True)
    val_df = pd.read_json('./data/datasets/LACD-biclassification/train-test-divide/val.jsonl', lines=True)


    if args.data_augmentation == "None":
        pass
    elif args.data_augmentation == "balance":
        train_df = data_augmentation(train_df, args.data_augmentation)
        train_df = balance_dataframe(train_df)
    else:
        train_df = data_augmentation(train_df, args.data_augmentation)

    # train_df에 'case_idx' 컬럼을 추가하고 모든 값을 0으로 설정
    train_df['case_idx'] = 0
    test_df['case_idx'] = 0
    val_df['case_idx'] = 0

    original_train_df = train_df.copy()
    # case_multiplier가 1보다 클 경우에만 아래의 코드를 실행
    if "case" in args.method and case_multiplier > 1:
        # 1부터 case_multiplier-1까지 반복
        for i in range(1, case_multiplier):
            # train_df를 복제
            temp_df = original_train_df.copy()
            # 복제한 데이터프레임의 'case_idx' 값을 i로 설정
            temp_df['case_idx'] = i
            # 복제한 데이터프레임을 원래 데이터프레임에 붙여넣기
            train_df = pd.concat([train_df, temp_df], ignore_index=True)

    
    train_dataset = NLIDataset(train_df, tokenizer, max_length=MAX_TOKEN_LENGTH, method=args.method)
    test_dataset = NLIDataset(test_df, tokenizer, max_length=MAX_TOKEN_LENGTH, method=args.method)
    val_dataset = NLIDataset(val_df, tokenizer, max_length=MAX_TOKEN_LENGTH, method=args.method)

    print(f"Token limit for {args.model}: {tokenizer.model_max_length}")
    train_dataset.print_label_counts()
    test_dataset.print_label_counts()
    val_dataset.print_label_counts()
    
    training_args = TrainingArguments(
        output_dir=f'./outputs/LACD-cross/small-fine-tune/{tag}',  # output directory
        num_train_epochs=epoch_num,                          # total number of training epochs
        per_device_train_batch_size=4,               # batch size for training
        per_device_eval_batch_size=4,                # batch size for evaluation
        warmup_steps=500,                            # number of warmup steps for learning rate scheduler
        weight_decay=0,
        logging_dir=f'./outputs/LACD-cross/small-fine-tune/{tag}',  # directory for storing logs
        # logging_steps=10,
        eval_strategy="steps",                       # evaluation strategy
        eval_steps=20,                               # evaluation interval
        save_strategy="steps",                       # save strategy to match eval steps
        save_steps=20,                               # save interval matching eval steps
        save_total_limit=1,                          # only keep the best model
        load_best_model_at_end=True,                 # load the best model at the end
        metric_for_best_model="eval_roc_auc",           # metric to use for model selection
        greater_is_better=True,                     # True if a higher metric value is better
        report_to="tensorboard"                      # report to TensorBoard
    )
    if args.mode == "train":
        print('train')

        writer = SummaryWriter()

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained # type: ignore
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,           # evaluation dataset
            compute_metrics=compute_metrics,
            callbacks=[TensorBoardCallback(writer)]
        )

        trainer.train()

        test_results = trainer.evaluate(eval_dataset=test_dataset)

         # 결과를 저장
        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)

        # 결과를 출력
        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

        # TensorBoard에 기록
        writer.add_scalar("Test/F1", test_f1)
        writer.add_scalar("Test/Accuracy", test_accuracy)
        writer.add_scalar("Test/ROC_AUC", test_roc_auc)

        if args.model_save_path == "None":
            path = f"./data/models/LACD-cross/{tag}"
        else:
            path = args.model_save_path
        
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model, path+"/model.pth")
        tokenizer.save_pretrained(path)
            
    elif args.mode == "test":
        print("evaluate")
        model = torch.load(f"./data/models/LACD-cross/{tag}/model.pth")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained # type: ignore
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
            compute_metrics=compute_metrics
        )
        eval_result = trainer.evaluate()

        # Print test results
        test_f1 = eval_result.get("eval_f1", 0)
        test_accuracy = eval_result.get("eval_accuracy", 0)
        test_roc_auc = eval_result.get("eval_roc_auc", 0)
        test_precision = eval_result.get("eval_recall", 0)
        test_recall = eval_result.get("eval_precision", 0)

        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test precision Score: {test_precision:.1%}")
        print(f"Test recall Score: {test_recall:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")

        # 예측 수행
        # predictions = trainer.predict(test_dataset)
        # pred_labels = predictions.predictions.argmax(axis=1)  # 예측된 레이블
        # true_labels = predictions.label_ids  # 실제 레이블
        
        # # 틀린 instance 저장
        # incorrect_instances = []
        
        # for i in range(len(true_labels)):
        #     if pred_labels[i] != true_labels[i]:  # 예측이 틀린 경우
        #         incorrect_instances.append({
        #             'index': i,
        #             'article1': test_df.iloc[i]['article1'],  # test 데이터에서 첫 번째 문장
        #             'article2': test_df.iloc[i]['article2'],  # test 데이터에서 두 번째 문장
        #             'true_label': int(true_labels[i]),               # 실제 레이블
        #             'predicted_label': int(pred_labels[i])      # 예측된 레이블
        #         })
        
        # # 틀린 instance를 jsonl 파일로 저장
        # import json
        # incorrect_file_path = f"./outputs/LACD-cross/small-fine-tune/{tag}/incorrect_instances.jsonl"
        # with open(incorrect_file_path, 'w') as f:
        #     for instance in incorrect_instances:
        #         f.write(json.dumps(instance, ensure_ascii=False) + '\n')
    
    if args.method == "case-augmentation" or args.method == "case-concat-augmentation":
        case_cache_end()
