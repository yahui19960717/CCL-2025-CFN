##加载LTP https://hf-mirror.com/LTP/base2
import os
import torch
from ltp import LTP
import numpy as np
import json
import tqdm
# 加载HanLP依存句法分析模型z
def get_ws(text, word_positions):
    new_text = []
    num = 0
    for word in word_positions:
        start = word['start']
        end = word['end']
        new_text.append(text[start:end+1])
        num += 1
    sen = " ".join(new_text)
    # 创建ID与分词的映射
    id_to_word = {i: new_text[i] for i in range(len(new_text))}
    # ID到字符位置的映射
    id_to_position = {i: word_positions[i] for i in range(len(word_positions))}
    return [sen, id_to_position, id_to_word]

def read_cfn(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for ins in data:
        text = ins['text']
        ws = ins['word']
        sen, id_to_position, id_to_word = get_ws(text, ws)
        ins['ws_text'] = sen
    #     ins['id2words'] = id_to_word
    #     ins['id2position'] = id_to_position
    print(f'*****{data_path} 共有{len(data)}条数据******')
    return data

def write_json(data, file_out):
    with open(file_out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
 

def dep_parsing(data, file_out=None):
    ltp = LTP('../base2')
    if torch.cuda.is_available():
        ltp.to("cuda")
    for ins in tqdm.tqdm(data):
        text_split = ins['ws_text'].split()
        import pdb; pdb.set_trace()
        dep_model = ltp.pipeline(text_split, tasks=["pos", "dep"])
        assert len(dep_model["dep"]['head']) == len(dep_model["dep"]['label']) == len(text_split) == len(dep_model["pos"])
        ins['dep_head'] = dep_model["dep"]['head']
        ins['dep_label'] = dep_model["dep"]['label']
        ins['pos'] = dep_model["pos"]
    
    if file_out:
        write_json(data, file_out)
    
    return data
if __name__ == '__main__':
    ### 获取依存句法分析结果
    train = '../dataset/cfn-train.json'
    dev = '../dataset/cfn-dev.json'
    test = '../dataset/cfn-test-A.json'
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = read_cfn(train)
    dev_data = read_cfn(dev)
    test_data = read_cfn(test)
    ### 获得成分句法分析结果
    print(f'开始处理训练集' )
    comp_parsing(train_data, file_out='../dataset/cfn-dep/cfn-train-comp-pos.json')
    exit()
    ### 获取依存句法分析结果
    # print(f'开始处理训练集' )
    # dep_parsing(train_data, file_out='../dataset/cfn-dep/cfn-train-dep-pos.json')
    # print(f'开始处理开发集')
    # dep_parsing(dev_data, file_out='../dataset/cfn-dep/cfn-dev-dep-pos.json')
    # print(f'开始处理测试集')
    # dep_parsing(test_data, file_out='../dataset/cfn-dep/cfn-test-A-dep-pos.json')

 