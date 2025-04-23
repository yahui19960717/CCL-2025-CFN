import torch
import codecs
import json
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False, ws=False, use_embedding=False, embed_file=None):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        self.ws = ws
        self.use_embedding = use_embedding
        self.embed_file = embed_file
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.label2idx = {}
        self.label2cls = {}
        self.num_labels = 0
        for k, line in enumerate(self.ori_labels):
            frame_name = line["frame_name"]
            self.label2cls[frame_name] = k
            if frame_name not in self.label2idx:
                self.label2idx[frame_name] = {}
            for i, fes in enumerate(line["fes"]):
                self.label2idx[frame_name][fes["fe_name"]] = i
            if self.num_labels < len(line["fes"]):
                self.num_labels = len(line["fes"])
        self.max_cls = len(self.label2idx)
        
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
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        target_cls = self.label2cls[d1["frame"]]
        target_text = d1['text'][target[0]-1:target[1]]

        label = []
        for line in d1["cfn_spans"]:
            if line["end"] + 1 < target[0]:
                label.append([line["start"] + 1, line["end"] + 1])
            elif line["start"] + 1 > target[1]:
                label.append([line["start"] + 3, line["end"] + 3])
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        ws_label = None if self.ws is False else [self.ws2idx[line] for line in d1["ws"]]

        word_indices = None if self.use_embedding is False else [self.word2idx_emb[target_text]]
        return input_ids, attention_mask, target, label, sentence_id, target_cls, ws_label, word_indices
 

if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset("./dataset/cfn-train.json",
                      "./dataset/frame_info.json",
                      tokenizer=tokenizer)

    dataset[0]





