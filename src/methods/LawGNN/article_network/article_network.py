import torch
import json
import tqdm
import csv

class ArticleNetwork:
    def __init__(self, laws_path="./data/database/laws.csv", law_link_path = "./data/database/law_link_20240930_duplicate_eliminate.jsonl"):
        # Dictionary to store the network (adjacency list)
        self.article_network = {}
        # Dictionary to store the article key to index mapping
        self.article_key_to_idx = {}
        # List to store all article keys
        self.all_article_keys = []
        
        # Load all articles from laws_html.jsonl
        self._load_nodes(laws_path)
        self._load_edges(law_link_path)

    def _load_nodes(self, laws_path):
        # TODO: article 수정해야 함.
        # print("Loading article nodes...")

        reader = csv.DictReader(open(laws_path))
        for row in reader:
            article_key = row['article_title']
            article_key = article_key.replace("·", "ㆍ")

            self.all_article_keys.append(article_key)

        # Create a mapping from article key to index
        self.article_key_to_idx = {key: idx for idx, key in enumerate(self.all_article_keys)}
        # print(f"Loaded {len(self.all_article_keys)} articles as nodes.")
    
    def add_article_node(self, article):
        from src.utils.utils import article_key_function
        article_key = article_key_function(article)
        article_idx = len(self.all_article_keys)
        self.all_article_keys.append(article_key)

        # 맨 마지막에 넣음
        self.article_key_to_idx[article_key]=article_idx
        return article_idx
    
    def _load_edges(self, law_link_path):
        # """Load edges (connections) from law_link JSONL file"""
        # print("Loading article network edges...")
        skipped_edge_cnt = 0
        with open(law_link_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f):
                link_data = json.loads(line)
                source = link_data['source_key']
                target = link_data['target_key']
                
                if (source not in self.article_key_to_idx.keys()) or (target not in self.article_key_to_idx.keys()):
                    skipped_edge_cnt = skipped_edge_cnt +1
                    continue

                # Add source -> target connection
                if source not in self.article_network:
                    self.article_network[source] = []
                self.article_network[source].append(target)
        # print(f"Loaded edges for {len(self.article_network)} articles.")
        # print(f"{skipped_edge_cnt} edges are skipped.")

    def find_connected_articles(self, article_key):
        """Return the list of articles connected to a given article key"""
        return self.article_network.get(article_key, [])

    def create_edge_index(self):
        """Create edge index from the article network"""
        edge_index = [[], []]  # List to store the source and target of edges


        no_found_cnt = 0
        # Ensure all edges in the article network are added to the edge_index
        for source, targets in self.article_network.items():
            if source in self.article_key_to_idx:
                for target in targets:
                    if target in self.article_key_to_idx:
                        # Add source -> target
                        # 실험해볼 만한 것 역순으로 넣어야 전파가 된다.
                        edge_index[1].append(self.article_key_to_idx[source])
                        edge_index[0].append(self.article_key_to_idx[target])
                        
                        edge_index[0].append(self.article_key_to_idx[source])
                        edge_index[1].append(self.article_key_to_idx[target])                        
                    else:

                        no_found_cnt += 1
            else:
                no_found_cnt += 1


        # Convert to a tensor
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        return edge_index_tensor

if __name__ == "__main__":
    # Initialize the ArticleNetwork class
    article_network = ArticleNetwork()
        
    # Find connected articles for a specific key
    print(article_network.find_connected_articles("특정범죄 가중처벌 등에 관한 법률 제2조"))
    
    # Create edge index and print it
    print(article_network.create_edge_index())