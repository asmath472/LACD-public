# prompt-based case generator

def case_gen_prompt(article: str):
      
    return f"""
<법률 사례 생성>
설명: 주어진 법률 조문에 대해서, 사람들이 그 법률 조문을 더욱 쉽게 이해할 수 있도록 법률이 적용되는 예시 사례를 반드시 한국어로 생성해줘. 

다음은 예시들이야:

Article: 형법 제276조 (체포, 감금, 존속체포, 존속감금) ①사람을 체포 또는 감금한 자는 5년이하의 징역 또는 700만원 이하의 벌금에 처한다.

Case: 피고인과 피해자 B(여, 42세)은 연인 관계이다. 1. 피고인은 2021. 6. 15. 19:55경 광주 북구 C건물 D호에 피해자와 함께 입실하여, 피해자가 다른 남자와 계속해서 연락을 한다는 이유로 시비하였다. 이후 같은 해 6. 16. 02:00경 피고인은 모텔 복도로 피해자가 나가는 것을 보고 피해자를 뒤따라와 손을 들어 피해자를 때리려고 하는 행동을 하고, 팔을 붙잡아 당기는 유형력을 행사하여 다시 모텔 안으로 데리고 들어가 나무젓가락을 부러뜨려 피해자를 위협하는 방법으로 피해자를 같은 날 05:30경까지 위 모텔에서 못나게 하여 약 3시간 동안 감금하였다. 2. 피고인은 2021. 6. 20. 00:00경 광주 북구 E건물 F호에 피해자와 함께 숙박하였다. 피고인은 위 모텔에서 피해자와 남자관계 문제로 언쟁을 하였고, 술에 취해 피고인이 횡설수설하여 귀가를 하려고 하자 자신의 휴대전화를 창문 밖에 집어 던지고 피해자에게 전화기 수화기를 들이밀고, 손을 들어 피해자를 때릴 듯이 유형력을 행사하며 “내가 가만히 있을 것 같냐, 난 끝까지 간다, 나 징역 갈 마음으로 이곳에 왔다. 너 일하는 병원에 내 동생들도 보낼 꺼다”라고 협박하여 피해자를 약 7시간 동안 모텔에서 나가지 못하게 하여 감금하였다.

----

Article: 형법 제297조(강간) 폭행 또는 협박으로 사람을 강간한 자는 3년 이상의 유기징역에 처한다.

Case: 피고인은 2019. 9. 22. 21:00경 인터넷 채팅을 통해 알게 된 피해자 B(가명, 여, 33세)가 술에 취한 피고인을 경산시 C모텔 객실에 데려다주게 되었음을 기회로, 객실을 나가려는 피해자를 침대로 밀어 넘어뜨린 다음 ‘하지 말라’고 말하며 벗어나려는 피해자의 몸을 힘으로 누르며 억지로 피해자의 하의와 속옷을 벗긴 다음 ‘내 여자로 만들어야 한다’고 말을 하며 피해자의 성기에 피고인의 성기를 삽입하였다. 이로써 피고인은 피해자를 강간하였다.

----

Article: 형법 제298조(강제추행) 폭행 또는 협박으로 사람에 대하여 추행을 한 자는 10년 이하의 징역 또는 1천500만원 이하의 벌금에 처한다.

Case: [전제사실] 피고인은 2020. 12. 8. 22:00경 광주 북구 B에 있는 피고인이 운영하는 ‘C’에서 손님인 피해자 D(여, 33세) 및 피해자의 일행 E과 술을 마시게 되었고, 피해자 및 위 E과 인근에 있는 노래방으로 가려고 하였으나 노래방이 문을 닫아 가지 못하자, 광주 북구 F에 있는 G마트 두암점으로 아이스크림을 사러 가게 되었다. 피고인은 2020. 12. 8. 23:50경 위 G마트 두암점 내 아이스크림 진열대 옆에서, E이 아이스크림을 고르는 모습을 지켜보고 있던 피해자의 뒤쪽으로 다가가, “가슴이 크네”라고 말하면서 갑자기 피해자의 가슴을 손으로 만지고, 계속하여 피해자의 엉덩이를 손으로 만졌다. 이로써 피고인은 피해자를 강제로 추행하였다.

----

Article: 형법 제319조(주거침입, 퇴거불응) ①사람의 주거, 관리하는 건조물, 선박이나 항공기 또는 점유하는 방실에 침입한 자는 3년 이하의 징역 또는 500만원 이하의 벌금에 처한다. 

Case: 피고인은 2019. 2. 8.부터 2019. 4. 17.까지 피해자 B 주식회사(이하 ‘피해자 회사’라고 한다)에서 근무한 사람으로, 피해자 회사에서 퇴사하면서 임금 정산 등의 문제로 피해자 회사의 대표이사 L와 분쟁이 있어 왔다. 피고인은 2019. 8. 14. 04:24경 서산시 M에 있는 피해자 회사의 가스 충전소 앞에 이르러 피해자 회사의 위법사항을 찾는다는 이유로 출입금지 표지판이 부착되어 있는 차량 진입로를 통해 가정용·용기 충전소 안까지 들어가 피해자 회사가 관리하는 건조물에 침입하였다.

----

Article: 형법 제136조(공무집행방해) ①직무를 집행하는 공무원에 대하여 폭행 또는 협박한 자는 5년 이하의 징역 또는 1천만원 이하의 벌금에 처한다.

Case: 피고인은 2021. 9. 10. 15:25경 부산 연제구 B에 있는 C 앞길에서 함께 술을 마셨던 D와 택시에 탑승하여 모텔에 함께 가자고 제안하였으나 이를 거절한다는 이유로 D에게 폭행을 가하였다. 피고인은 2021. 9. 10. 15:53경 같은 장소에서 위와 같은 폭행에 대한 112신고를 받고 현장에 출동한 부산연제경찰서 소속 경사 E로부터 사건 경위를 파악하기 위해 인적사항을 요구받자 이를 거부하며 “씹할놈아 니는 뭐고. 일처리나 똑바로 해라. 그런식으로 하면 내가 니 죽인다.”라고 소리치면서 손으로 E의 머리채를 잡아 흔드는 등 폭행하였다. 이로써 피고인은 112신고 처리에 관한 경찰관의 정당한 직무집행을 방해하였다.

----
<질문>

Article: {article}

Case: 
"""

import openai
import pandas as pd
from tqdm import tqdm
import json

from src.utils.utils import article_key_function
# ./data/database/generated_case_cache/prompt-base-full-law/case_cache.jsonl
def case_cache_start(cache_path= "./data/database/generated_case_cache/prompt-base/case_cache.jsonl"):
    global cache_df

        # Load the cache DataFrame
    try:
        cache_df = pd.read_json(cache_path, lines=True)
    except ValueError:
        # If the cache file is empty or doesn't exist, create an empty DataFrame
        cache_df = pd.DataFrame(columns=["article", "case"])

    # return cache_df

def case_cache_end(cache_path = "./data/database/generated_case_cache/prompt-base/case_cache.jsonl"):
    global cache_df
   
    # Save the updated cache DataFrame back to the file
    cache_df.to_json(cache_path, lines=True, orient="records", force_ascii=False)


def generate_case(model, client, article: str, use_case_cache=True, article_key=None, case_idx = 0) -> str:

    global cache_df
    if article_key == None:
        article_key = article_key_function(article)
        # print(article_key)
    if use_case_cache:
        # Check if the article exists in the cache

        if "-" in article_key:
            alter_article_key = article_key.split("-")[1]+"-"+article_key.split("-")[0]
        
            matching_rows = cache_df[(cache_df["article_key"] == article_key) | (cache_df["article_key"] == alter_article_key)]
        else:
            matching_rows = cache_df[cache_df["article_key"] == article_key]

        # 0은 아님.
        if len(matching_rows) > case_idx:
            # Return the cached case if a match is found
            return matching_rows.iloc[case_idx]["case"]
        else:
            # 이러면 Generate
            pass

    if model == None:
        # 임시로 생성하게 하기. 나중에는 바꿔야 함.
        model = "gpt-4o-mini"
        client = openai.OpenAI()


    
    # Generate a new case using the LLM
    case_query = case_gen_prompt(article)
    generated_case_text = client.chat.completions.create(model=model, messages = [{"role": "user", "content": case_query}])
    generated_case_text = generated_case_text.choices[0].message.content


    if use_case_cache:
        # Append the new article and generated case to the cache DataFrame
        new_entry = pd.DataFrame([{"article": article, "case": generated_case_text, "article_key": article_key}])
        cache_df = pd.concat([cache_df, new_entry], ignore_index=True)

        
    return generated_case_text

# 전체 case-cache 를 generation 하는 코드
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import json
    from tqdm import tqdm
    import numpy as np
    # argparse 설정
    parser = argparse.ArgumentParser(description="case cache generator")

    # 모델 경로 동적으로 입력 받기
    parser.add_argument("--machine_num", type=int, help="machine num for distributed case-generation", default=-1)

    args = parser.parse_args()


    # 모든 법률에 대한 case 저장
    if args.machine_num == -1:
        case_cache_path = "./data/database/generated_case_cache/prompt-base-full-law/case_cache.jsonl"
    else:
        case_cache_path = "./data/database/generated_case_cache/prompt-base-full-law/case_cache_{}.jsonl".format(args.machine_num)

    buffer_df = pd.DataFrame(columns=["article", "case", "article_key"])

    # model_path = "Qwen/Qwen2-7B-Instruct"
    # model_path = "/mnt/disk1/anseon2001/transformers-cache/models--Qwen--Qwen2-72B-Instruct/snapshots/fddbbd7b69a1fd7cf9b659203b37ae3eb89059e1"

    model_path = "gpt-4o-mini"
    client = openai.OpenAI()

    # openai_api_key = "EMPTY"
    # openai_api_base = "http://localhost:8001/v1"

    # client = openai.OpenAI(
    #     api_key=openai_api_key,
    #     base_url=openai_api_base,
    # )


    # 모든 cache 를 다운로드
    laws_csv = "./data/database/laws.csv"
    laws_df = pd.read_csv(laws_csv)

    # 데이터프레임을 10등분
    num_splits = 10
    split_dfs = np.array_split(laws_df, num_splits)

    # machine_num이 주어진 범위 내에 있는지 확인
    if args.machine_num < 0 or args.machine_num >= num_splits:
        raise ValueError(f"Invalid machine_num: {args.machine_num}. Must be between 0 and {num_splits-1}.")

    # machine_num에 해당하는 부분을 가져오기
    laws_df = pd.DataFrame(split_dfs[args.machine_num])

    idx = 0

    print("laws_df len:{}".format(len(laws_df)))

    for _, row in tqdm(laws_df.iterrows()):
        
        article = row['contents']
        # 캐싱하지 않음
        generated_case_text  = generate_case(model = model_path, client=client, article=article, use_case_cache=False)
        article_key_text = article_key_function(article)
        new_entry = pd.DataFrame([{"article": article, "case": generated_case_text, "article_key": article_key_text}])
        buffer_df = pd.concat([buffer_df, new_entry], ignore_index=True)

        idx = idx + 1

        if idx % 1000 == 0:
            with open(case_cache_path, 'a', encoding="utf-8") as f:
                for index, row in buffer_df.iterrows():
                    # 행을 딕셔너리로 변환
                    row_dict = row.to_dict()
                    # json 문자열을 파일에 쓰기
                    f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')

            buffer_df.drop(buffer_df.index, inplace=True)

    # 남은 것 저장
    with open(case_cache_path, 'a', encoding="utf-8") as f:
        for index, row in buffer_df.iterrows():
            # 행을 딕셔너리로 변환
            row_dict = row.to_dict()
            # json 문자열을 파일에 쓰기
            f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')