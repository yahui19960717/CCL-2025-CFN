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

def vote_task2(list_data, file_path, num_vote=3):
    all_list = []
    final_results   = []
    dic_results = defaultdict(list)
    for data in list_data:
        data = read_json(data)
        for item in data:
            all_list.append(tuple(item))
 
    all_list_count = Counter(all_list) 
    for key, value in all_list_count.items():
        if value >= num_vote:
            final_results.append(list(key))
    
    write_json(final_results, file_path)
    return final_results



if __name__ == "__main__":
    # list_data = ["../dataset/baseline-roberta-large/A_task2_test.json",
    #              "../dataset/target_embedding_ws/A_task2_test.json",
    #              "../dataset/target_embedding/A_task2_test.json",
    #              "../dataset/ws_add/A_task2_test.json",
    #              "../dataset/baseline-ws/A_task2_test.json",
    #             #  "../dataset/baseline/A_task2_test.json",          
    #             ]
    # vote_task2(list_data, "../dataset/vote1-6/A_task2_test.json")  # >=3

    list_data = ["../dataset/baseline-roberta-large/A_task2_test.json",
                "../dataset/seed77/A_task2_test.json",
                 "../dataset/seed777/A_task2_test.json",
                 ]
    vote_task2(list_data, "../dataset/vote2-3/A_task2_test.json", num_vote=2)  # >=3