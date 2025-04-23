import json

with open('../dataset/cfn-dep/cfn-train-dep-pos.json', 'r', encoding='utf8') as f:
    data = json.load(f)
for ins in data:
     

    print('-'*100)
    import pdb; pdb.set_trace()