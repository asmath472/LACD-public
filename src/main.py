import chromadb
import tqdm
import torch
from src.utils.utils import article_key_function


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*past_key_values.*")
    
    import argparse

    # argparse 설정
    parser = argparse.ArgumentParser(description="legal contradiction detector")

    # 모델 경로 동적으로 입력 받기
    parser.add_argument("--biencoder_model_path", type=str, help="Biencoder model path", default="./data/models/LACD-bi/kbb-baseline")
    parser.add_argument("--crossencoder_model_path", type=str, help="Crossencoder model path", default="./data/models/LACD-cross/kbb-baseline")

    parser.add_argument("--laws_csv_path", type=str, help="Path to the laws.csv file", default="./data/database/laws.csv")
    parser.add_argument("--classification_method", type=str, choices=["cosine", "classification_model"], help="Choose between 'cosine' similarity or 'classification_model' for contradiction detection", default="cosine")
    parser.add_argument("--chroma_db_name", type=str, required=True, help="Name of the Chroma DB where encodings will be stored")
    
    parser.add_argument("--biencoder_top_k", type=int, default=10, help="binary_encoder_top_k")
    parser.add_argument("--biencoder_method", type=str, choices=["baseline", "caseaug"], help="Choose between 'baseline' similarity or 'caseaug' for cross enocder method", default="baseline")
    parser.add_argument("--crossencoder_top_k", type=int, default=10, help="cross_encoder_top_k")
    parser.add_argument("--crossencoder_method", type=str, choices=["baseline", "caseaug"], help="Choose between 'baseline' similarity or 'caseaug' for cross enocder method", default="baseline")

    # retrieval method
    parser.add_argument("--retrieval_method", type=str, choices=["bi-only", "cross-only", "hybrid","tfidf", "bm25"], help="retrieval method. bi-encoder only, cross-encoder only, tfidf, bm25, or hybrid", default="bi-only")
    parser.add_argument("--crossencoder_index_method", type=str, choices=["none", "vanilla", "gcn", "graphsage", "gat"], help="index usage for cross encoder. none means do not use index. vanilla means use index w/o GNNs.", default="none")

    parser.add_argument("--mode", type=str, choices=["inference", "test-benchmark"], help="benchmark test or inference once", default="inference")

    # crossencoder vector index 사용하는지도 필요함.
    # GNN method 적어야 함.

    args = parser.parse_args()

    # 사용 예시
    biencoder_model_path = args.biencoder_model_path
    biencoder_top_k = args.biencoder_top_k
    crossencoder_model_path = args.crossencoder_model_path
    crossencoder_top_k = args.crossencoder_top_k
    laws_csv_path = args.laws_csv_path
    classification_method = args.classification_method
    # biencoder_chroma_db_name = args.biencoder_chroma_db_name
    # crossencoder_chroma_db_name = args.crossencoder_chroma_db_name

    chroma_db_name = args.chroma_db_name

    crossencoder_index_method = args.crossencoder_index_method


    if biencoder_model_path in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
        from src.encoders.bi_encoder.retriever.openai_binary_retriever import openai_biencoder_retriever as binary_retriever
    else:
        from src.encoders.bi_encoder.retriever.binary_retriever import binary_retriever

    if "noLM" in crossencoder_model_path:
        from src.encoders.cross_encoder.retriever.cross_retriever import noLM_cross_retriever as cross_retriever

    else:
        from src.encoders.cross_encoder.retriever.cross_retriever import cross_retriever



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 21대 국회에서 임기만료로 폐기. 의안번호 26497. 
    # https://pal.assembly.go.kr/napal/flexer/out/eaa8a5b8-4f57-4d79-8d6f-5a8530aac28e.hwp.files/Sections1.html
    # 22대 국회에서 이달희 의원 대표발의, 국회 본회의 통과
    article_to_check = """아동ㆍ청소년의 성보호에 관한 법률 제11조의2(아동ㆍ청소년성착취물을이용한협박ㆍ강요) ①아동ㆍ청소년 성착취물을 이용하여 사람을 협박한 자는 5년 이상의 유기징역에 처한다. ②제1항에 따른 협박으로 사람의 권리행사를 방해하거나 의무없는 일을 하게한 자는 7년 이상의 유기징역에 처한다. ③제1항과 제2항의 미수범은 처벌한다. ④상습적으로 제1항 및 제2항의 죄를 범한자는 그죄에 대하여 정하는 형의 2분의1까지 가중한다."""
    # article_to_check = """성폭력범죄의 처벌 등에 관한 특례법 제14조의2(허위영상물 등의 반포등) ① 사람의 얼굴ㆍ신체 또는 음성을 대상으로 한 촬영물ㆍ영상물 또는 음성물(이하 이 조에서 “영상물등”이라 한다)을 영상물등의 대상자의 의사에 반하여 성적 욕망 또는 수치심을 유발할 수 있는 형태로 편집ㆍ합성 또는 가공(이하 이 조에서 “편집등”이라 한다)한 자는 7년 이하의 징역 또는 7천만원 이하의 벌금에 처한다.② 제1항에 따른 편집물ㆍ합성물ㆍ가공물(이하 이 항에서 “편집물등”이라 한다) 또는 복제물(복제물의 복제물을 포함한다. 이하 이 항에서 같다)을 반포등을 한 자 또는 제1항의 편집등을 할 당시에는 영상물등의 대상자의 의사에 반하지 아니한 경우에도 사후에 그 편집물등 또는 복제물을 영상물등의 대상자의 의사에 반하여 반포등을 한 자는 7년 이하의 징역 또는 7천만원 이하의 벌금에 처한다.③ 제1항에 따른 영상물등과 제2항에 따른 편집물등을 소지ㆍ구입ㆍ저장 또는 시청한 자는 3년 이하의 징역 또는 3천만원 이하의 벌금에 처한다.④ 영리를 목적으로 영상물등의 대상자의 의사에 반하여 정보통신망을 이용하여 제2항의 죄를 범한 자는 7년 이하의 징역에 처한다.⑤ 상습으로 제1항부터 제4항까지의 죄를 범한 때에는 그 죄에 정한 형의 2분의 1까지 가중한다."""
    # article_to_check = """계엄사령부 포고령 제1조 국회와 지방의회, 정당의 활동과 정치적 결사, 집회, 시위 등 일체의 정치활동은 금지된다."""
    
    


    from src.methods.LawGNN.article_network.article_network import ArticleNetwork
    from src.utils.utils import article_key_function

    # print("a"+article_key_function(article_to_check)+"a")
    article_network = ArticleNetwork()
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)

    # bi-encoder 혹은 cross-encoder 에서 caseaug method 가 있다면
    if args.biencoder_method == "caseaug" or args.crossencoder_method == "caseaug":
        from src.methods.case_augmentation.prompt import case_cache_start, case_cache_end
        case_cache_start(cache_path="./data/database/generated_case_cache/prompt-base-full-law/case_cache.jsonl")
        # case_cache_start(cache_path="./data/database/generated_case_cache_with_real_cases/prompt-base-full-law/case_cache.jsonl")




    if args.mode =="inference":
        if args.retrieval_method == "bi-only":
            article_vector, top_contradictions = binary_retriever(biencoder_model_path, laws_csv_path, chroma_db_name, article_to_check, classification_method, biencoder_top_k, case_augmentation_method=args.biencoder_method)
            top_contradictions = [t.split("\n[CASE]\n")[0] for t in top_contradictions]

        elif args.retrieval_method == "cross-only":
            client = chromadb.PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)

            # 이미 저장된 인코딩이 있으면 불러오기
            chroma_collection = client.get_or_create_collection("quickstart")
            encoded_laws = chroma_collection.get(include=["documents"])
            law_documents = encoded_laws["documents"]
            top_contradictions = cross_retriever(article_to_check, 
            law_documents, 
            crossencoder_model_path, 
            top_k=crossencoder_top_k,
            method = args.crossencoder_method, 
            article_network=article_network, 
            crossencoder_index_db=chroma_db_name, 
            index_method=crossencoder_index_method,
            )

        elif args.retrieval_method == "hybrid":
            article_vector, top_contradictions = binary_retriever(biencoder_model_path, laws_csv_path, chroma_db_name, article_to_check, classification_method, top_k=biencoder_top_k, case_augmentation_method=args.biencoder_method)

            top_contradictions = cross_retriever(article_to_check, 
            top_contradictions, 
            crossencoder_model_path,
            top_k=crossencoder_top_k,
            method = args.crossencoder_method,
            article_network=article_network,
            crossencoder_index_db=chroma_db_name,
            index_method=crossencoder_index_method,
            article_vector=article_vector
            )

        # TF-IDF
        elif args.retrieval_method == "tfidf":
            from src.encoders.bi_encoder.retriever.classical_retriever import find_top_contradictions_with_tfidf as tfidf_retriever
            top_contradictions = tfidf_retriever(article_to_check, chroma_db_name, top_k=biencoder_top_k, method=args.biencoder_method)
        
        # BM25
        elif args.retrieval_method == "bm25":
            from src.encoders.bi_encoder.retriever.classical_retriever import find_top_contradictions_with_bm25 as bm25_retriever
            top_contradictions = bm25_retriever(article_to_check, chroma_db_name, top_k=biencoder_top_k, method=args.biencoder_method)

        else:
            assert(0)
        

        # verbose?
        print("모순된 법률 article:")
        for idx, contradiction in enumerate(top_contradictions, 1):
            print(f"{idx}: {contradiction}")
    
    else:
        import json
        from tqdm import tqdm

        # 결과를 저장할 리스트
        result_list = []

        # 기존의 rows 읽기 작업
        rows = []
        with open("./data/datasets/LACD-biclassification/train-test-divide/test.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                row = json.loads(line.strip())  # 각 줄을 JSON으로 파싱
                rows.append(row)

        true_count = 0
        false_count = 0

        idx = 0
        for row in tqdm(rows):
            idx = idx+1
            article1 = row["article1"]
            article_to_check = article1

            if args.retrieval_method == "bi-only":
                article_vector, top_contradictions = binary_retriever(biencoder_model_path, laws_csv_path, chroma_db_name, article_to_check, classification_method, biencoder_top_k, case_augmentation_method=args.biencoder_method)
                top_contradictions = [t.split("\n[CASE]\n")[0] for t in top_contradictions]

            elif args.retrieval_method == "cross-only":
                client = chromadb.PersistentClient(path="./data/database/chroma_db/" + chroma_db_name)
                chroma_collection = client.get_or_create_collection("quickstart")
                encoded_laws = chroma_collection.get(include=["documents"])
                law_documents = encoded_laws["documents"]
                top_contradictions = cross_retriever(article_to_check, 
                    law_documents, 
                    crossencoder_model_path, 
                    top_k=crossencoder_top_k,
                    method=args.crossencoder_method, 
                    article_network=article_network, 
                    crossencoder_index_db=chroma_db_name, 
                    index_method=crossencoder_index_method
                )

            elif args.retrieval_method == "hybrid":
                article_vector, top_contradictions = binary_retriever(biencoder_model_path, laws_csv_path, chroma_db_name, article_to_check, classification_method, top_k=biencoder_top_k, case_augmentation_method=args.biencoder_method)

                top_contradictions = cross_retriever(article_to_check, 
                    top_contradictions, 
                    crossencoder_model_path,
                    top_k=crossencoder_top_k,
                    method=args.crossencoder_method,
                    article_network=article_network,
                    crossencoder_index_db=chroma_db_name,
                    index_method=crossencoder_index_method,
                    article_vector=article_vector
                )
            # TF-IDF
            elif args.retrieval_method == "tfidf":
                from src.encoders.bi_encoder.retriever.classical_retriever import find_top_contradictions_with_tfidf as tfidf_retriever
                top_contradictions = tfidf_retriever(article_to_check, chroma_db_name, top_k=biencoder_top_k, method=args.biencoder_method)
            
            # BM25
            elif args.retrieval_method == "bm25":
                from src.encoders.bi_encoder.retriever.classical_retriever import find_top_contradictions_with_bm25 as bm25_retriever
                top_contradictions = bm25_retriever(article_to_check, chroma_db_name, top_k=biencoder_top_k, method=args.biencoder_method)


            articles = top_contradictions

            # article_to_check와 articles의 쌍을 result_list에 추가
            result_list.append({"article_to_check": article_to_check, "articles": articles})

        # result_list를 jsonl 파일로 저장
        with open("./outputs/retrieval_results/{0}_{1}_{2}{3}.jsonl".format(args.biencoder_method,crossencoder_index_method ,args.crossencoder_method,"_noLM" if "noLM" in crossencoder_model_path else ""), 'w', encoding='utf-8') as outfile:
            for result in result_list:
                json.dump(result, outfile, ensure_ascii=False)
                outfile.write('\n')  # 각 결과를 한 줄에 저장

