import os
import json

def calculate_statistics(directory):
    files_data = []
    total_word_sum = 0
    total_files = 0
    all_answers_count = {'true': 0, 'false': 0}

    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                article_word_sum = 0
                article_count = 0
                answer_count = {'true': 0, 'false': 0}
                
                for line in f:
                    data = json.loads(line)
                    article1_words = len(data['article1'].split())
                    article2_words = len(data['article2'].split())
                    article_word_sum += (article1_words + article2_words)
                    article_count += 1
                    
                    if data['answer'] == True:
                        answer_count['true'] += 1
                    elif data['answer'] == False:
                        answer_count['false'] += 1
                
                avg_word_length_sum = article_word_sum / article_count if article_count > 0 else 0
                files_data.append({
                    'file': filename,
                    'average_word_length_sum': avg_word_length_sum,
                    'answer_count': answer_count,
                })
                
                total_word_sum += article_word_sum
                total_files += article_count
                all_answers_count['true'] += answer_count['true']
                all_answers_count['false'] += answer_count['false']

    overall_avg_word_length_sum = total_word_sum / total_files if total_files > 0 else 0

    return files_data, all_answers_count, overall_avg_word_length_sum


directory_path = './data/datasets/LACD-biclassification/train-test-divide'
files_data, all_answers_count, overall_avg_word_length_sum = calculate_statistics(directory_path)

# 결과 출력
for file_data in files_data:
    print(f"파일: {file_data['file']}")
    print(f"평균 단어 길이 합: {file_data['average_word_length_sum']}")
    print(f"True의 수: {file_data['answer_count']['true']}, False의 수: {file_data['answer_count']['false']}")
    print("")

print(f"전체 파일들의 평균 article 1 과 article 2 의 평균 word 길이 합: {overall_avg_word_length_sum}")
print(f"전체 True의 수: {all_answers_count['true']}, 전체 False의 수: {all_answers_count['false']}")
