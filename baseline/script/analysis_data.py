
## Author : @yhliu
## TIME   : 2025.4.11
## Content: 分析frame/train/dev/test数据
### @Frame: 框架个数、框架名称、框架的平均框架元素数量、框架的平均词汇单元数量
### @CFN: 句子个数、涉及框架数，框架分布，平均句子长度，句子中涉及的框架元素个数，框架元素分布
import json
from collections import defaultdict
def read_frame(file):
    num_fes = 0
    dic_fes = {}
    with open(file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        # dict_keys(['frame_name', 'frame_ename', 'frame_def', 'fes'])
        # fes:['fe_name', 'fe_abbr', 'fe_ename', 'fe_def']
        # note: 一个frame_name可能对应多个fes
        max_fes = 0
        temp_fes = ''
        for item in data:
            num_fes += len(item["fes"])
            if len(item["fes"]) > max_fes:
                max_fes = len(item["fes"])
                temp_fes = item["frame_name"]
                temp_fes_list = item["fes"]
            for fe in item["fes"]:
                if fe['fe_name'] not in dic_fes:
                    dic_fes[fe['fe_name']] = 1
                else:
                    dic_fes[fe['fe_name']] += 1    

    print(f'number of frames: {len(data)}')
    print(f'number of different fes: {len(dic_fes)}') # labels
    print(f'number of fes: {num_fes:,}') 
    print(f'average number of fes: {num_fes/len(data):.2f}')
    print(f'max number of fes: {max_fes}')
    print(f'frame name with max fes: {temp_fes}')
    # print(f'fes with max number: {temp_fes_list}')
    return data

def get_word_segmentation(text, word_positions):
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

def get_target(text, target):
    target_list = []
    for i in range(len(target)):
        temp = target[i]
        target_list.append([temp['start'], temp['end'], text[temp['start']:(temp['end']+1)]])
    return target_list

def get_cfn_span(cfn_spans):
    cfn_span_list = []
    for item in cfn_spans:
        cfn_span_list.append([item['start'], item['end'], item['fe_name']])
    return cfn_span_list

def judge_span_cross(span_list):
    for i in range(len(span_list)):
        for j in range(i+1, len(span_list)):
            if span_list[i][0] < span_list[j][0] and span_list[i][1] > span_list[j][1]:
                return True
    return False    

def read_cfn(file, file_out=None):
    with open(file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        multi_target_num, single_target_num, no_target_num, all_target_num = 0, 0, 0, 0
        num_words, num_spans = 0, 0
        dic_frame = defaultdict(int)
        new_data = []
        for item in data:
            # ['sentence_id', 'cfn_spans', 'frame', 'target', 'text', 'word']
            temp_dic = {}
            temp_dic['sentence_id'] = item['sentence_id']
            temp_dic['frame'] = item['frame']
            dic_frame[item['frame']] += 1
            target = get_target(item['text'], item['target'])
            temp_dic['target'] = target
            temp_dic['cfn_span_list'] = get_cfn_span(item['cfn_spans'])
            num_spans += len(temp_dic['cfn_span_list'])
            temp_dic['ws_sen'] = get_word_segmentation(item['text'], item['word']) #[sen, id_to_position, id_to_word]
            num_words += len(temp_dic['ws_sen'][0].split())
            if len(temp_dic['target']) > 1:
                multi_target_num += 1
                # import pdb; pdb.set_trace()
            elif len(target) == 1:
                single_target_num += 1
            elif len(target) == 0:
                no_target_num += 1
            all_target_num += len(target)
            # if judge_span_cross(cfn_span_list): # 如果存在交叉的span，则输出
                # import pdb; pdb.set_trace()
            new_data.append(temp_dic)
    # print(f'{multi_target_num:10,}, {single_target_num:10,}, {no_target_num:10,}, {all_target_num:10,}')
    print(f'{len(data):10,}, {len(dic_frame):10,}, {num_words/len(data):.2f}, {len(dic_frame)/len(data):.2f}, {num_spans/len(data):.2f}')
    # print(sorted(dic_frame.items(), key=lambda x: x[1], reverse=True))
    # print(dic_frame)
    # print(f'number of data: {len(data):,}')
    # print(f'number of frames: {len(dic_frame):,}')
    # print(f'number of multi-target: {multi_target_num:,}')
    # print(f'number of single-target: {single_target_num:,}')
    # print(f'number of no-target: {no_target_num:,}')
    # print(f'number of all-target: {all_target_num:,}')
    if file_out:
        with open(file_out, "w", encoding="utf-8-sig") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=0)
    return data

def analyze_frequency_distribution(frame_freq):
    # 定义频率区间,闭区间
    intervals = [(0,0), (1, 15), (16, 50), (51, 100), (101, float('inf'))]
    interval_counts = defaultdict(int)
    interval_frames = defaultdict(list)
    
    for frame, freq in frame_freq.items():
        for start, end in intervals:
            if start <= freq <= end:
                interval_key = f"{start}-{end if end != float('inf') else '∞'}"
                interval_counts[interval_key] += 1
                interval_frames[interval_key].append((frame, freq))
                break
    
    # 打印统计结果
    print("\n频率区间分布统计:")
    print("-" * 50)
    for interval, count in interval_counts.items():
        print(f"频率区间 {interval}: {count}个框架")
        print("示例框架:")
        # 打印每个区间的前5个框架及其频率
        for frame, freq in sorted(interval_frames[interval], key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {frame}: {freq}")
        print()


def analysis_frame_train(frame, cfn_train):
    frame_data = read_frame(frame)
    frame_dic = defaultdict(int)
    for item in frame_data:
        frame_dic[item['frame_name']] = 0
    cfn_train_data = read_cfn(cfn_train)
    for item in cfn_train_data:
        temp = item['frame']
        frame_dic[temp] += 1
    sorted_frame_dic = sorted(frame_dic.items(), key=lambda x: x[1], reverse=True)
    dic_new = {key:value for key, value in sorted_frame_dic}
    count_sen, count_frame = 0, 0
    for key, value in dic_new.items():
        count_sen += value
    print(f'number of sentences: {count_sen:,}, average number of frames: {count_sen/len(frame_dic):.2f}')
    analyze_frequency_distribution(dic_new)


if __name__=="__main__":
    # 所有的句子都有target
    cfn_train = "../dataset/cfn-train.json"
    cfn_dev   = "../dataset/cfn-dev.json"
    cfn_test  = "../dataset/cfn-test-A.json" 
    frame = "../dataset/frame_info.json"  
    # 分析每个框架在train中出现的频率
    analysis_frame_train(frame, cfn_train)
    exit()


    deald_train = "../dataset/dealed/train.json"
    deald_dev   = "../dataset/dealed/dev.json"
    deald_test  = "../dataset/dealed/test-A.json"

    # 2. 读取cfn数据,dealed
    cfn_train_data = read_cfn(cfn_train, deald_train)
    cfn_dev_data = read_cfn(cfn_dev, deald_dev)
    cfn_test_data = read_cfn(cfn_test)

    exit()
    # 1. 读取和分析frame数据
    # frame = "/data/yhliu/CFN2024/CFSP/LLM/data/frame_info.json"
    frame_data = read_frame(frame)

