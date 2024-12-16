import json

from sympy import false, true
from src.utils.utils import article_key_function
import os
from tqdm import tqdm
if __name__ == "__main__":
    
    rows = []
    top_k = 5
    biencoder_top_k=10

    with open("./data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            row = json.loads(line.strip())  # 각 줄을 JSON으로 파싱
            rows.append(row)

    with open("./data/datasets/LACD-retrieval/missed_pairs.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            row = json.loads(line.strip())  # 각 줄을 JSON으로 파싱
            rows.append(row)


    rows = [{"article1": article_key_function(r["article1"]), "article2": article_key_function(r["article2"]), "answer": r["answer"]} for r in rows]

    missed_pairs = []
    seen_pairs = set()  # 중복된 쌍을 확인하기 위한 set
    for tags in ["baseline_none_baseline", "caseaug_none_baseline" ,"baseline_gat_baseline", "caseaug_gat_baseline"]:

    # for tags in ["caseaug_gcn_caseaug"]:
        print(f"==\ntags: {tags}")
        
        results = []

        with open(f"./outputs/retrieval_results/biencoder-top-{biencoder_top_k}-qwen/{tags}.jsonl", 'r', encoding='utf-8') as result_file:
            for line in result_file:
                row = json.loads(line.strip())  # 각 줄을 JSON으로 파싱
                results.append(row)

        checked_articles = set()

        true_positive = 0
        false_positive = 0
        blind_negative = 0
        false_negative = 0
        blind_positive = 0
        count = 0

        for result in results:
            article_to_check = result['article_to_check']
            
            # 한번씩만 검증함
            if article_to_check in checked_articles:
                continue

            checked_articles.add(article_to_check)

            articles = result['articles']
            articles = [a for a in articles if article_key_function(article_to_check) != article_key_function(a)]

            if len(articles) == 0:
                matching_rows = [r for r in rows if article_key_function(r["article1"]) == article_key_function(article_to_check) or article_key_function(r["article2"]) == article_key_function(article_to_check)]
                
                for r in matching_rows:
                    # 하나라도 True 이면 잘못 대답한 것이므로 false
                    if r['answer'] is True:
                        false_negative = false_negative + 1
                        break
                
                if r['answer'] is False:
                    blind_negative = blind_negative + 1

            else:
                # print(article_key_function(article_to_check), [article_key_function(a) for a in articles])
                for a in articles[:top_k]:
                    # rows 중에서 "article1"과 "article2"가 각각 article1, a로 일치하는 행을 찾음
                    matching_rows = [r for r in rows if (article_key_function(r["article1"]) == article_key_function(article_to_check) and article_key_function(r["article2"]) == article_key_function(a)) or (article_key_function(r["article2"]) == article_key_function(article_to_check) and article_key_function(r["article1"]) == article_key_function(a))]

                    # assert(len(matching_rows) < 2)

                    if len(matching_rows) == 0:
                        pair = (article_key_function(article_to_check), article_key_function(a))
                        reverse_pair = (pair[1], pair[0])
                        
                        # 중복 확인: pair가 이미 seen_pairs에 있는지 확인
                        if pair not in seen_pairs and reverse_pair not in seen_pairs:
                            missed_pairs.append({"article1": article_to_check, "article2": a, "answer": None})
                            seen_pairs.add(pair)  # 새로운 쌍을 seen_pairs에 추가
                            blind_positive += 1

                    # 매칭된 행들에 대해 row['answer']가 True 또는 False인지 확인하여 카운트
                    elif matching_rows[0]['answer'] is True:
                        true_positive += 1

                    elif matching_rows[0]['answer'] is False:
                        false_positive += 1
                    else:
                        # 아직 채점되지 않은것
                        pass
                        # assert(0)

        query_num = len(checked_articles)

        # 대답하지 않은 query 수
        no_answer_query = blind_negative + false_negative
        # 대답한 query 수
        answer_query = query_num - no_answer_query
        print(f"For {query_num} queries total, retriever answered {answer_query} of queries and did not answered {no_answer_query} of queries.")
        print(f"For no_answer_query, false_negative: {false_negative}, blind_negative: {blind_negative}")
        print(f"For answered query, true_positive: {true_positive}, false_positive: {false_positive}, blind_positive:{blind_positive}")

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative+blind_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"recall at {top_k}: {recall}")
        print(f"precision at {top_k}: {precision}")

    # 중복 없는 missed_pairs를 missed_pairs.jsonl로 저장
    with open("missed_pairs.jsonl", 'w', encoding='utf-8') as outfile:
        for pair in missed_pairs:
            json.dump(pair, outfile, ensure_ascii=False)
            outfile.write('\n')