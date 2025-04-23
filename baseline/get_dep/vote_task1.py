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

def vote_task1(list_data, file_path):
    all_list = []
    final_results   = []
    dic_results = defaultdict(list)
    for data in list_data:
        data = read_json(data)
        for item in data:
            dic_results[item[0]].append(item[1])
    print(len(dic_results))
    dic_final = {}
    for key, value in dic_results.items():
        elements = Counter(value).most_common(1)
        dic_final[key] = elements[0][0]
    
    for item in data:
        final_results.append([item[0], dic_final[item[0]]])
    write_json(final_results, file_path)
    return final_results



if __name__ == "__main__":
    # list_data = ["../dataset/baseline-roberta-large/A_task1_test.json",
    #              "../dataset/target_embedding_ws/A_task1_test.json",
    #              "../dataset/target_embedding/A_task1_test.json",
    #              "../dataset/ws_add/A_task1_test.json",
    #              "../dataset/baseline-ws/A_task1_test.json",
    #             #  "../dataset/baseline/A_task1_test.json",          
    #             ]
    # vote_task1(list_data, "../dataset/vote1-6/A_task1_test.json")

    list_data = ["../dataset/we-falsefreeze/A_task1_test.json",
                "../dataset/seed77/A_task1_test.json",
                "../dataset/seed777/A_task1_test.json",
                ]
    vote_task1(list_data, "../dataset/vote2-3/A_task1_test.json")
 