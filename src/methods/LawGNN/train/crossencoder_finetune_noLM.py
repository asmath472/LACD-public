# 작동 방식
# python ./src/methods/LawGNN/train/crossencoder_finetune_noLM.py --tag kbb-caseaug-noLM-gat --gnn_method gat --chroma_db_name kbb-caseaug --epoch 3

import torch
from torch.utils.tensorboard import SummaryWriter # type: ignore
from transformers import Trainer, TrainingArguments
import numpy as np
import argparse
from chromadb import PersistentClient
from tqdm import trange
import json
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.methods.LawGNN.gnn_architecture import  NoCrossGATv2EncoderModel, NoCrossGCNEncoderModel, NoCrossSAGEEncoderModel, NoCrossVanillaEncoderModel, NoCrossVanillaGATv2EncoderModel, NoCrossVanillaSAGEEncoderModel
from src.utils.utils import article_key_function, seed_everything, SEED
from src.utils.encoder.utils import compute_metrics, MAX_TOKEN_LENGTH , TensorBoardCallback
from src.utils.gnn.crossencoder_utils import GNNNLIDataset
import warnings
import os

# Function to load data from jsonl
def load_dataset(jsonl_file, article_network):
    dataset = []
    key_error_cnt = 0
    # Ensure vector_tensor and edge_index_tensor are moved to CPU

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            article1 = data['article1']
            article2 = data['article2']
            try:
                article1_idx = article_network.article_key_to_idx[article_key_function(article1)]
                article2_idx = article_network.article_key_to_idx[article_key_function(article2)]
            except:
                key_error_cnt += 1
                continue
            
            label = 1 if data['answer'] else 0  # True -> 1, False -> 0
            dataset.append((article1_idx, article2_idx, None, None, label))

            

            
    return dataset

def load_chromadb(chroma_db_name, edge_index_tensor):
    client = PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
    chroma_collection = client.get_or_create_collection("quickstart")


    # Load all contents from Chroma DB
    all_contents = chroma_collection.get(include=["embeddings", "documents"])
    embeddings = all_contents.get("embeddings", [])
    documents = all_contents.get("documents", [])

    if len(embeddings) > 0:
        embedding_size = len(embeddings[0])
    else:
        raise ValueError("No embeddings found in ChromaDB.")
    
    collection_dict = {}
    no_key_count = 0
    for i in range(len(embeddings)):  # Use documents length as the reference
        article_key = article_key_function(documents[i])
        try:
            article_idx = article_network.article_key_to_idx[article_key]
        except:
            no_key_count += 1
            continue  # Skip if key not found
        entry = {
            "embedding": np.array(embeddings[i]) if i < len(embeddings) else None,
            "article_key": article_key
        }
        collection_dict[article_idx] = entry

    # Model initialization

    # vectors 만들기. tensor 로 만들어야 함.
    max_node_idx = edge_index_tensor.max().item()
    vectors_list = []
    for idx in range(int(max_node_idx + 1)):
        if idx in collection_dict.keys():
            vectors_list.append(torch.tensor(collection_dict[idx]["embedding"], dtype=torch.float32).to(device))
        else:
            vectors_list.append(torch.zeros(embedding_size, dtype=torch.float32).to(device))
    vector_tensor = torch.stack(vectors_list).to(device)
    return vector_tensor, embedding_size

# Example execution
if __name__ == "__main__":
    seed_everything(SEED)
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="train or test", default="train")
    parser.add_argument("--model_save_path", type=str, help="a path for saving model. do not use None as the name", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")
    parser.add_argument("--chroma_db_name", type=str, help="Name of the Chroma DB where encodings will be stored")
    parser.add_argument("--vanilla_chroma_db_name", type=str, help="Name of the Chroma DB where encodings will be stored", default= "None")
    parser.add_argument("--gnn_chroma_db_name", type=str, help="Name of the Chroma DB where encodings will be stored", default= "None")
    parser.add_argument("--gnn_method", type=str, help="Name of GNN method", default="gcn")
    parser.add_argument("--epoch", type = int, default=3)

    parser.add_argument("--no_cross", type=bool, default=False)

    args = parser.parse_args()

    chroma_db_name = args.chroma_db_name
    gnn_method = args.gnn_method
    epoch_num = args.epoch


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if case_augmentation_method == "case-augmentation" or case_augmentation_method == "case-concat-augmentation":
    #     from src.methods.case_augmentation.prompt import case_cache_start, case_cache_end
    #     case_cache_start()

    if args.vanilla_chroma_db_name == "None":
        if not os.path.exists("./data/database/chroma_db/" + chroma_db_name):
            print("Chroma DB does not exist")
            assert(0)
    else:
        if not os.path.exists("./data/database/chroma_db/" + args.vanilla_chroma_db_name):
            print("Chroma DB does not exist")
            assert(0)

    # Initialize Chroma DB client before creating the model
    # Article Network 만들기
    article_network = ArticleNetwork()
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)

    if args.vanilla_chroma_db_name == "None":
        vector_tensor, embedding_size = load_chromadb(chroma_db_name, edge_index_tensor)
        vanilla_vector_tensor = vector_tensor
        gnn_vector_tensor = vector_tensor
    else:
        chroma_dbs = [args.vanilla_chroma_db_name, args.gnn_chroma_db_name]
        chroma_db_load_list = [load_chromadb(c, edge_index_tensor) for c in chroma_dbs]
        vanilla_vector_tensor, embedding_size = chroma_db_load_list[0]
        gnn_vector_tensor, _ = chroma_db_load_list[1]


    # 만약 둘 다 쓴다면
    # 

    # GNN Bi-Encoder 모델 초기화 using dynamic embedding size
    if gnn_method == "gcn":
        model = NoCrossGCNEncoderModel(in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
        
    elif gnn_method == "graphsage":
        model = NoCrossSAGEEncoderModel(in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
        
    elif gnn_method == "gat":
        model = NoCrossGATv2EncoderModel(in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)

    elif gnn_method == "vanilla":
        model = NoCrossVanillaEncoderModel(in_channels=embedding_size, out_channels=embedding_size, vector_tensor=vector_tensor, edge_index_tensor=edge_index_tensor).to(device)

    elif gnn_method == "gat-hybrid":
        model = NoCrossVanillaGATv2EncoderModel(in_channels=embedding_size, out_channels=embedding_size, vanilla_vector_tensor=vanilla_vector_tensor, gnn_vector_tensor = gnn_vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    
    elif gnn_method == "graphsage-hybrid":
        model = NoCrossVanillaSAGEEncoderModel(in_channels=embedding_size, out_channels=embedding_size, vanilla_vector_tensor=vanilla_vector_tensor, gnn_vector_tensor = gnn_vector_tensor, edge_index_tensor=edge_index_tensor).to(device)
    else:
        print("improper GNN methods!")
        assert(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Load datasets
    train_dataset = GNNNLIDataset(load_dataset('./data/datasets/LACD-biclassification/train-test-divide/train.jsonl', article_network))
    # we do not need to multiply cases for val, test datasets
    val_dataset = GNNNLIDataset(load_dataset('./data/datasets/LACD-biclassification/train-test-divide/val.jsonl', article_network))
    test_dataset = GNNNLIDataset(load_dataset('./data/datasets/LACD-biclassification/train-test-divide/test.jsonl', article_network))

    training_args = TrainingArguments(
        output_dir=f'./outputs/LACD-cross/gnns/{args.tag}',  # output directory
        num_train_epochs=epoch_num,                          # total number of training epochs
        per_device_train_batch_size=4,               # batch size for training
        per_device_eval_batch_size=4,                # batch size for evaluation
        warmup_steps=500,                            # number of warmup steps for learning rate scheduler
        weight_decay=0,
        logging_dir=f'./outputs/LACD-cross/gnns/{args.tag}',  # directory for storing logs
        learning_rate=1e-3,
        eval_strategy="epoch",                       # evaluation strategy
        eval_steps=20,                               # evaluation interval
        save_strategy="epoch",                       # save strategy to match eval steps
        save_steps=20,                               # save interval matching eval steps
        save_total_limit=1,                          # only keep the best model
        load_best_model_at_end=True,                 # load the best model at the end
        metric_for_best_model="eval_roc_auc",           # metric to use for model selection
        greater_is_better=True,                     # True if a higher metric value is better
        report_to="tensorboard"                      # report to TensorBoard
    )

    writer = SummaryWriter()

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained # type: ignore
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback(writer)]
    )

    if args.mode == "train":
        print("Training...")
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
            path = f"./data/models/LACD-cross/gnns/{args.tag}"
        else:
            path = args.model_save_path
        
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model, path+"/model.pth")


    elif args.mode == "test":
        print("evaluate")
        model = torch.load(f"./data/models/LACD-cross/gnns/{args.tag}/model.pth")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained # type: ignore
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
            compute_metrics=compute_metrics
        )
        test_results = trainer.evaluate()

        # Print test results
        test_f1 = test_results.get("eval_f1", 0)
        test_accuracy = test_results.get("eval_accuracy", 0)
        test_roc_auc = test_results.get("eval_roc_auc", 0)
        test_precision = test_results.get("eval_recall", 0)
        test_recall = test_results.get("eval_precision", 0)

        print(f"Test F1 Score: {test_f1:.1%}")
        print(f"Test precision Score: {test_precision:.1%}")
        print(f"Test recall Score: {test_recall:.1%}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Test ROC AUC: {test_roc_auc:.1%}")
    


    
    # if case_augmentation_method == "case-augmentation" or case_augmentation_method == "case-concat-augmentation":
    #     case_cache_end()

