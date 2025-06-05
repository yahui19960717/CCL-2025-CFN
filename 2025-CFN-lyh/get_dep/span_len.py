import json
import numpy as np

def calculate_length_distribution(lengths):
    """
    计算长度列表的分布统计信息。

    Args:
        lengths (list of int): 包含所有论元或实体长度的列表。

    Returns:
        dict: 包含最小值、最大值、平均值、中位数以及指定百分比长度的字典。
    """
    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "average": 0,
            "median": 0,
            "90th_percentile": 0,
            "95th_percentile": 0,
        }
    lengths_array = np.array(lengths)
    results = {
        "min": np.min(lengths_array),
        "max": np.max(lengths_array),
        "average": np.mean(lengths_array),
        "median": np.median(lengths_array),
        "90th_percentile": np.percentile(lengths_array, 90),
        "95th_percentile": np.percentile(lengths_array, 95),
    }
    return results

# 示例用法 (文本列表)
# entity_texts_example = ['我 爱 北京', '天安门', '的', '中心', '中华人民共和国']
# distribution_text = calculate_length_distribution_from_text(entity_texts_example)
# print("Length Distribution (from text):", distribution_text)
def read_cfn(file):
    with open(file, 'r') as f:
        span_list = []
        dic_span_wide = {}
        data = json.load(f)
        for ins in data:
            text = ins['cfn_spans']
            for key in text:
                start =key['start']
                end = key['end']
                wide = end-start+1
                span_list.append(wide)
                dic_span_wide[wide] = [ins['text'][start:end+1], ins['text']]
    distribution_text = calculate_length_distribution(span_list)
    print("Length Distribution (from text):", distribution_text)
    # max_wide = 0
    # for key in dic_span_wide:
    #     if key > max_wide:
    #         max_wide = key
    #         max_span = dic_span_wide[key][0]
    #         max_text = dic_span_wide[key][1]
    # print(max_wide)
    # print(max_span)
    # print(max_text)
    # print/(sorted(dic_span_wide.keys()))
    return data




if __name__ == '__main__':
    train = '../dataset/cfn-train.json'
    dev = '../dataset/cfn-dev.json'
    # test = '../dataset/cfn-test-A.json'

    data_train = read_cfn(train)
    data_dev = read_cfn(dev)
    # data_test = read_cfn(test)

