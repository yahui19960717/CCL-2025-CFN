import json
from depparsing import get_ws
    
def read_cfn(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

def get_ws_bies(data, file):
    for item in data:
        list_ws = []
        word = item['word']
        text = item['text']
        sen, id_to_position, id_to_word = get_ws(text, word)
        item['ws_text'] = sen
        for w in word:
            start = w['start']
            end = w['end']
            if start == end:
                list_ws.append("S")
            elif start == end-1:
                list_ws.append("B")
                list_ws.append("E")
            else:
                list_ws.append("B")
                for i in range(start+1, end):
                    list_ws.append("M")
                list_ws.append("E")
        item['ws'] = list_ws
 
        
    write_json(data, file)
    return item



    
if __name__ == '__main__':
    train = '../dataset/cfn-train.json'
    dev = '../dataset/cfn-dev.json'
    test = '../dataset/cfn-test-A.json'

    ws_data_train = "../dataset/cfn_ws/cfn-train-ws.json"
    ws_data_dev = "../dataset/cfn_ws/cfn-dev-ws.json"
    ws_data_test = "../dataset/cfn_ws/cfn-test-A-ws.json"

    data_train = read_cfn(train)
    data_dev = read_cfn(dev)
    data_test = read_cfn(test)

    get_ws_bies(data_train, ws_data_train)
    get_ws_bies(data_dev, ws_data_dev)
    get_ws_bies(data_test, ws_data_test) 