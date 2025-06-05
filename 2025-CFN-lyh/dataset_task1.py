import torch
import codecs
import json
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False, pos_file=None, dep_file=None, ws=False, use_embedding=False, embed_file=None):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.ws = ws
        self.use_embedding = use_embedding
        # 对文本进行分词
        self.tokenizer = tokenizer
        self.pos_file = pos_file
        self.dep_file = dep_file
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        ## 框架文件
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        ## 添加词性文件和dep文件
        if pos_file:
            with codecs.open(pos_file, 'r', encoding='utf8') as f:
                self.ori_pos = json.load(f)
            self.idx2pos = []
            self.pos2idx = {}
            for i, line in enumerate(self.ori_pos):
                self.idx2pos.append(line)
                self.pos2idx[line] = i
        if dep_file:
            with codecs.open(dep_file, 'r', encoding='utf8') as f:
                self.ori_dep = json.load(f)
            self.idx2dep = []
            self.dep2idx = {}
            for i, line in enumerate(self.ori_dep):
                self.idx2dep.append(line)
                self.dep2idx[line] = i

        self.idx2label = []
        self.label2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.idx2label.append(line["frame_name"])
            self.label2idx[line["frame_name"]] = i

        self.num_labels = len(self.idx2label)

        
        if self.ws:
            self.ws2idx = {"B": 1, "M": 2, "E": 3, "S": 4}
            self.idx2ws= ["B", "M", "E", "S"]

        if self.use_embedding:  
            self.word2idx_emb = {}
            self.idx2word_emb = []
            self.embeddings = []
            with codecs.open(embed_file, 'r', encoding='utf8') as f:
                json_data = json.load(f)
            # add pad
            self.word2idx_emb["<PAD>"] = len(self.idx2word_emb)
            self.idx2word_emb.append("<PAD>")
            self.embeddings.append([0.0] * 300)
            # add unk
            self.word2idx_emb["<UNK>"] = len(self.idx2word_emb)
            self.idx2word_emb.append("<UNK>")
            self.embeddings.append([0.0] * 300)

            for w_temp, vector_temp in json_data.items():
                if w_temp in self.word2idx_emb:
                    continue
                self.word2idx_emb[w_temp] = len(self.idx2word_emb)
                self.idx2word_emb.append(w_temp)
                self.embeddings.append(vector_temp)

        

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        # '中央处理港澳事务，从来都是从战略和全局高度加以考量，从来都以国家和港澳的根本利益、长远利益为出发点和落脚点。'
        # '去年，一般贸易进出口总值为一千一百七十亿美元，增长百分之十四点五；加工贸易持续稳定发展，去年进出口总值一千六百九十八点一亿美元，增长百分之十五点八，占全国外贸总值的百分之五十二点二，比上年提高一点六个百分点。'

        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        target_text = d1['text'][target[0]-1:target[1]]
        label = self.label2idx[d1["frame"]]
        sentence_id = d1["sentence_id"]
        pos = None if self.pos_file is None else d1["pos"]
        dep = None if self.dep_file is None else d1["dep_label"]
        ws_label = None if self.ws is False else [self.ws2idx[line] for line in d1["ws"]]
        if self.use_embedding: # 应该是词在embedding中的index
            # word_indices = [self.word2idx_emb[i] for i in d1['ws_text'].split()]
            word_indices = [self.word2idx_emb[target_text]]
        return input_ids, attention_mask, target, label, sentence_id, pos, dep, ws_label, word_indices


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]




