## 获得所有的词性标签和依存标签，并存放在json文件中

import json
import codecs
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=0)

def read_json(file_path, f_pos, f_dep):
    with open(file_path, 'r', encoding='utf-8') as f:
        dic_pos = {}
        dic_dep = {}
        data = json.load(f)
        for ins in data:
            pos = ins['pos']
            dep = ins['dep_label']
            for i, p in enumerate(pos):
                if p not in dic_pos:
                    dic_pos[p] = 0
            for i, d in enumerate(dep):
                if d not in dic_dep:
                    dic_dep[d] = 0
                
    print(len(dic_pos), len(dic_dep))
    dic_pos['g'] = 0
    dic_pos['x'] = 0
    print(len(dic_pos), len(dic_dep))
    # print(json.dumps(sorted(dic_pos.items()), ensure_ascii=False, indent=0))
    # print(json.dumps(dic_dep, ensure_ascii=False, indent=0))
    write_json(dic_pos, f_pos)
    write_json(dic_dep, f_dep)
    return data
 


f_pos = '../dataset/cfn-dep/pos_labels.json'
f_dep = '../dataset/cfn-dep/dep_labels.json'
# data = read_json('../dataset/cfn-dep/cfn-train-dep-pos.json', f_pos, f_dep)
with codecs.open(f_pos, 'r', encoding='utf8') as f:
    ori_pos = json.load(f)

idx2pos = []
pos2idx = {}
for i, line in enumerate(ori_pos):
    idx2pos.append(line)
    pos2idx[line] = i

idx2dep = []
dep2idx = {}
with codecs.open(f_dep, 'r', encoding='utf8') as f:
    ori_dep = json.load(f)
    for i, line in enumerate(ori_dep):
        idx2dep.append(line)
        dep2idx[line] = i
import pdb;pdb.set_trace()

