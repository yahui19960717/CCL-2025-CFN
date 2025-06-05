
import json
from datetime import datetime
import time

def get_time_diff(start_time, end_time):
    # 计算时间差
    time_diff = end_time - start_time

    # 将时间差转换为秒数
    total_seconds = time_diff.total_seconds()

    # 计算时分秒
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # 输出结果
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"运行时间为: {formatted_time}")
def read_cfn_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"{file} has {len(data)} data")
    return data

def get_word_dic(data):
    word_dic = {}
    for item in data:
        for word in item['ws_text'].split():
             word_dic[word] = 0
    print(f"word_dic has {len(word_dic)} words")
    return word_dic
def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
def get_embedding(word_dic, embedding_file, embedding_picked):
    word_dic_embedding = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    embedding_dic = {}
    for line in lines:
        line = line.strip().split()
        embedding_dic[line[0]] = list(map(float, line[1:]))
    
    for word in word_dic:
        if word in embedding_dic:
            word_dic_embedding[word] = embedding_dic[word]
            
        else:
            word_dic_embedding[word] = [0.0] * 300
    
    write_json(word_dic_embedding, embedding_picked)
    print(f"word_dic_embedding has {len(word_dic_embedding)} words")

    
    

if __name__ == "__main__":
    ## 先查看所有cfn数据，获得分词
    start_time = datetime.now()
    embedding = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/cc.zh.300.vec"
    embedding_picked = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/cc.zh.300.cfn.vec"
    embedding_picked_includeB = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/cc.zh.300.cfnAB.vec"


    train_file = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/cfn-dep/cfn-train-dep-pos.json"
    dev_file = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/cfn-dep/cfn-dev-dep-pos.json"
    test_file = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/cfn-dep/cfn-test-A-dep-pos.json"
    testb_file = "/data/yhliu/The-3nd-Chinese-Frame-Semantic-Parsing/baseline/dataset/cfn-dep/cfn-test-B-dep-pos.json"

    train_data = read_cfn_data(train_file)
    dev_data = read_cfn_data(dev_file)
    test_data = read_cfn_data(test_file)
    testb_data = read_cfn_data(testb_file)

    word_dic_train = get_word_dic(train_data)
    word_dic_dev = get_word_dic(dev_data)   
    word_dic_test = get_word_dic(test_data)
    word_dic_testb = get_word_dic(testb_data)
    # 34500 三个数据集的merged词典
    #  39127 words 四个数据集
    merged_dict_operator = {**word_dic_train, **word_dic_dev, **word_dic_test, **word_dic_testb}
    
    # get_embedding(merged_dict_operator, embedding, embedding_picked)
    # 运行时间为: 00:03:52
    get_embedding(merged_dict_operator, embedding, embedding_picked_includeB)
    # 运行时间为: 00:03:32
    end_time = datetime.now()
    get_time_diff(start_time, end_time)
    