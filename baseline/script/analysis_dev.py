# @Author: yhliu
# @Date: 2025-04-15 20:00:00
# @Content: 分析dev数据

import json
from collections import defaultdict 

def read_data(file):
    with open(file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        print(f"file: {file}, len: {len(data)}")
        return data

def analysis_task1_dev(data, answer):
    # 分析task1的dev数据
    dic_gold = {item['sentence_id']:item['frame'] for item in data}
    gold_set = {(item['sentence_id'],item['frame']) for item in data}
    dic_pred = {item[0]:item[1] for item in answer}
    pred_set = {(item[0],item[1]) for item in answer}
    acc = len(gold_set & pred_set) / len(gold_set)
    print(f"acc: {acc}")
    # 统计每个框架的个数
    differ_dic = {}
    for item in dic_gold:
        if item in dic_pred and dic_gold[item] != dic_pred[item]:
            differ_dic[item] = [dic_gold[item], dic_pred[item]]

    # for item in differ_dic:

def check_devframe_in_train(dev_frame, train_data):
    dic_train, dic_dev = defaultdict(int), defaultdict(int)
    for item in train_data:     
        dic_train[item['frame']] += 1
    for item in dev_frame:
        dic_dev[item['frame']] += 1
    count = 0
    all_count = 0
    for key in dic_dev:
        if key not in dic_train:
            print(f"{key}: {dic_dev[key]}")
            count += dic_dev[key]   
        all_count += dic_dev[key]
    print(f"the sentence number of being in dev but not in train: {count / all_count :.2%}, {count}, {all_count}") 
    train_frame_set = {item['frame'] for item in train_data}
    dev_frame_set = {item['frame'] for item in dev_frame}
    cross_frame = dev_frame_set - train_frame_set
    print(f"the frame number of being in dev but not in train: {len(cross_frame)}")
    print(f"the frame number of being in train: {len(train_frame_set)}")
    print(f"the frame number of being in dev: {len(dev_frame_set)}")
    # the sentence number of being in dev but not in train: 2.78%, 64, 2300
    # the frame number of being in dev but not in train: 21
    # the frame number of being in train: 638
    # the frame number of being in dev: 368


def get_target(text, target):
    target_list = []
    for i in range(len(target)):
        temp = target[i]
        target_list.append([temp['start'], temp['end'], text[temp['start']:(temp['end']+1)]])
    return target_list

def check_devpredict_in_train(gold,pred, train_data):
    error_target = []
    gold_dic = {item['sentence_id']:item for item in gold}
    gold_set = {(item['sentence_id'],item['frame']) for item in gold}
    pred_set = {(item[0],item[1]) for item in pred}
    cross_set = pred_set - gold_set
    dic_train = {item['frame']:1 for item in train_data}
    count = 0
    for key in cross_set:
        assert key[0] in gold_dic.keys()
        if key[1] not in dic_train.keys():
            count += 1
        else:
            target_list = get_target(gold_dic[key[0]]['text'], gold_dic[key[0]]['target'])
            error_target.append(target_list)
            print(gold_dic[key[0]]['text'])
            print(f"id:{key[0]},  predict:: {key[1]} , gold:{gold_dic[key[0]]['frame']}, pred_in_train:{target_list}")
            print(f"\n")
    print(f"the error prediction number not in train: {count / len(pred_set) :.2%}, {count}, {len(pred_set)}")
    return error_target
def check_target_in_train(train_data, targets):
    dic_train = defaultdict(list)
    for key in train_data:
        target_list = get_target(key['text'], key['target'])
        if len(target_list) > 1:
            temp_target = "...".join([target_list[0][2], target_list[1][2]])
        else:
            temp_target = target_list[0][2]
        dic_train[temp_target].append(key['frame'])
    for target in targets:
        if target not in dic_train.keys():
            print(target)

if __name__ == "__main__":
    ## 看下dev中预测错误的target，在train中的框架分别是什么？
    train_data = read_data("../dataset/cfn-train.json")
    check_devpredict_in_train(dev_frame_gold, dev_frame_pred, train_data)
    exit()
    ## 看下dev中没有被预测出来的结果有哪些
    dev_frame_pred = read_data("../dataset/dev/task1_dev.json")
    dev_frame_gold = read_data("../dataset/cfn-dev.json")
    train_data = read_data("../dataset/cfn-train.json")
    check_devpredict_in_train(dev_frame_gold, dev_frame_pred, train_data)
    exit()

    # 查看dev的框架是否都在train中出现过，只有64个句子（21个框架）没有出现过
    data = read_data("../dataset/cfn-dev.json")
    answer = read_data("../dataset/dev/task1_dev.json")
    train_data = read_data("../dataset/cfn-train.json")
    check_devframe_in_train(data, train_data)


    exit()
    analysis_task1_dev(data, answer)









