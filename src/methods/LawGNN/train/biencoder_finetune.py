import torch
from torch_geometric.loader import DataLoader
import numpy as np
import argparse
from sklearn.metrics import f1_score, roc_auc_score
from chromadb import PersistentClient
from tqdm import trange

import json
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.methods.LawGNN.gnn_architecture import GCNBiEncoder, SAGEBiEncoder, GATv2BiEncoder
from src.utils.utils import article_key_function, seed_everything, SEED

import os


# Function to load data from jsonl
def load_dataset(jsonl_file, article_network):
    dataset = []
    key_error_cnt = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            article1 = data['article1']
            article2 = data['article2']
            try:
                article1_idx = article_network.article_key_to_idx[article_key_function(article1)]
                article2_idx = article_network.article_key_to_idx[article_key_function(article2)]
            except:
                key_error_cnt = key_error_cnt+1
                continue

            label = 1 if data['answer'] else 0  # True -> 1, False -> 0
            dataset.append((article1_idx, article2_idx, label))
    # print("key_error_cnt: {}".format(key_error_cnt))
    return dataset

# DataLoader 및 PyG Batch 구성
def collate_fn(batch):
    data_a_list = torch.tensor([item[0] for item in batch], dtype=torch.long)
    data_b_list = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)
    return data_a_list, data_b_list, labels


# Example execution
if __name__ == "__main__":
    seed_everything(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="train or test", default="train")
    parser.add_argument("--model_save_path", type=str, help="a path for saving model", default="None")
    parser.add_argument("--tag", type=str, help="tensorboard and output tag", default="None")
    parser.add_argument("--method", type=str, help="cosine or linear", default="cosine")
    parser.add_argument("--chroma_db_name", type=str, required=True, help="Name of the Chroma DB where encodings will be stored")
    parser.add_argument("--gnn_method", type=str, help="Name of GNN method", default = "gcn")
 
    args = parser.parse_args()

    chroma_db_name = args.chroma_db_name
    method = args.method
    gnn_method = args.gnn_method

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("./data/database/chroma_db/" + chroma_db_name):
        print("Chroma DB does not exists")
        assert(0)

    # Initialize Chroma DB client before creating the model
    client = PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
    chroma_collection = client.get_or_create_collection("quickstart")
    

    # Article Network 만들기
    article_network = ArticleNetwork()
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)


    # Load all contents from Chroma DB
    all_contents = chroma_collection.get(include=["embeddings", "documents"])
    embeddings = all_contents.get("embeddings", [])
    documents = all_contents.get("documents", [])

    if len(embeddings) > 0:
        embedding_size = len(embeddings[0])
    else:
        raise ValueError("No embeddings found in ChromaDB.")

    # print("embedding_size is", embedding_size)

    collection_dict = {}
    no_key_count = 0
    for i in range(len(embeddings)):  # Use documents length as the reference

        article_key = article_key_function(documents[i])
        try:
            article_idx = article_network.article_key_to_idx[article_key]
        except:
            no_key_count = no_key_count + 1
            continue  # Skip if key not found
        entry = {
            "embedding": np.array(embeddings[i]) if i < len(embeddings) else None,
            "article_key": article_key
        }
        collection_dict[article_idx] = entry

    print("no_key_count: {}".format(no_key_count))

    # GCN Bi-Encoder 모델 초기화 using dynamic embedding size
    if gnn_method == "gcn":
        model = GCNBiEncoder(in_channels=embedding_size, out_channels=embedding_size, method=method).to(device)
    elif gnn_method == "graphsage":
        model = SAGEBiEncoder(in_channels=embedding_size, out_channels=embedding_size, method=method).to(device)
    elif gnn_method == "gat":
        model = GATv2BiEncoder(in_channels=embedding_size, out_channels=embedding_size, method=method).to(device)
    else:
        print("improper GNN methods!")
        assert(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # vectors 만들기. tensor 로 만들어야 함.
    # Find the maximum node index in edge_index_tensor
    max_node_idx = edge_index_tensor.max().item()

    vectors_list = []
    for idx in range(int(max_node_idx + 1)):
        if idx in collection_dict.keys():
            vectors_list.append(torch.tensor(collection_dict[idx]["embedding"], dtype=torch.float32).to(device))
        else:
            vectors_list.append(torch.zeros(embedding_size, dtype=torch.float32).to(device))



    vector_tensor = torch.stack(vectors_list).to(device)

    # Load datasets
    train_dataset = load_dataset('./data/datasets/LACD-biclassification/train-test-divide/train.jsonl', article_network)
    val_dataset = load_dataset('./data/datasets/LACD-biclassification/train-test-divide/val.jsonl', article_network)
    test_dataset = load_dataset('./data/datasets/LACD-biclassification/train-test-divide/test.jsonl', article_network)

    batch_size = 128  # You can adjust the batch size as needed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if args.mode == "train":
        print("Training...")
        num_epochs = 30  # Adjust the number of epochs as needed
        best_val_f1 = 0.0
        best_model_state = None
        for epoch in trange(num_epochs):
            model.train()
            total_loss = 0
            for a_idx, b_idx, labels in train_loader:
                a_idx = a_idx.to(device)
                b_idx = b_idx.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, loss = model(vector_tensor, edge_index_tensor, a_idx, b_idx, labels=labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            # print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            # Evaluate on validation set
            model.eval()
            all_val_preds = []
            all_val_labels = []
            with torch.no_grad():
                for a_idx, b_idx, labels in val_loader:
                    a_idx = a_idx.to(device)
                    b_idx = b_idx.to(device)
                    labels = labels.to(device)
                    logits, _ = model(vector_tensor, edge_index_tensor, a_idx, b_idx)
                    if method == 'cosine':
                        preds = (logits > 0.5).long().squeeze()
                    elif method == 'linear':
                        preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
            # print(f"Epoch {epoch+1}, Validation F1 Score: {val_f1:.4f}")

            # Save the model if it has the best validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                # Save the best model
                if args.model_save_path == "None":
                    path = f"./data/models/LACD-bi/gnns/{args.tag}"
                else:
                    path = args.model_save_path
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(best_model_state, os.path.join(path, "model.pth"))
                # print(f"New best model saved at epoch {epoch+1} with Validation F1: {val_f1:.4f}")

        print("Training complete.")
        print(f"Best Validation F1 Score: {best_val_f1:.4f}")

        # Load the best model
        model.load_state_dict(best_model_state)

        # Evaluate on test set
        model.eval()
        all_test_preds = []
        all_test_labels = []
        with torch.no_grad():
            for a_idx, b_idx, labels in test_loader:
                a_idx = a_idx.to(device)
                b_idx = b_idx.to(device)
                labels = labels.to(device)
                logits, _ = model(vector_tensor, edge_index_tensor, a_idx, b_idx)
                if method == 'cosine':
                    preds = (logits > 0.5).long().squeeze()
                elif method == 'linear':
                    preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        # Calculate and print test metrics
        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
        print(f"Test F1 Score: {test_f1:.4f}")

    elif args.mode == "test":
        print("Testing...")
        # Load the saved model
        if args.model_save_path == "None":
            path = f"./data/models/LACD-bi/gnns/{args.tag}/model.pth"
        else:
            path = os.path.join(args.model_save_path, "model.pth")
        model.load_state_dict(torch.load(path))
        model.eval()
        all_test_preds = []
        all_test_labels = []
        with torch.no_grad():
            for a_idx, b_idx, labels in test_loader:
                a_idx = a_idx.to(device)
                b_idx = b_idx.to(device)
                labels = labels.to(device)
                logits, _ = model(vector_tensor, edge_index_tensor, a_idx, b_idx)
                if method == 'cosine':
                    preds = (logits > 0.5).long().squeeze()
                elif method == 'linear':
                    preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        # Calculate and print test metrics
        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
        print(f"Test F1 Score: {test_f1:.4f}")
