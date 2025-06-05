## 投票任务1
import json
from collections import Counter
from collections import defaultdict
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(len(data))
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        print(f"write {file_path}: length {len(data)}")
        json.dump(data, f, ensure_ascii=False, indent=0)

def vote_task3(list_data, file_path, num):
    all_list = []
    final_results   = []
    dic_results = defaultdict(list)
    for data in list_data:
        data = read_json(data)
        for item in data:
            all_list.append(tuple(item))
 
    all_list_count = Counter(all_list) 
    for key, value in all_list_count.items():
        if value >= num:
            final_results.append(list(key))
    
    write_json(final_results, file_path)
    return final_results



if __name__ == "__main__":


    list_data = ["../dataset/seed1/B_task3_test.json",
                "../dataset/seed77/B_task3_test.json",
                "../dataset/seed777/B_task3_test.json",
                ]
    vote_task3(list_data, "../dataset/submit/B_task3_test.json", num=2)

    exit()

    list_data = ["../dataset/baseline-roberta-large/A_task3_test.json",
                 "../dataset/target_embedding_ws/A_task3_test.json",
                 "../dataset/target_embedding/A_task3_test.json",
                 "../dataset/ws_add/A_task3_test.json",
                 "../dataset/baseline-ws/A_task3_test.json",
                #  "../dataset/baseline/A_task3_test.json",          
                ]
    vote_task3(list_data, "../dataset/vote1-6/A_task3_test.json")  # >=3
