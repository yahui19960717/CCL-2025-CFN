#!/usr/bin/python3
import os
import torch
import codecs
import json
from functools import partial
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from params2 import args
from model_task2 import Model


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False, ws=False, use_embedding=False, embed_file=None):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.ws = ws
        self.use_embedding = use_embedding
        self.embed_file = embed_file
        self.tokenizer = tokenizer
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

        pass

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        target_text = d1['text'][target[0]-1:target[1]]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]
        ws_label = None if self.ws is False else [self.ws2idx[line] for line in d1["ws"]]
        word_indices = None if self.use_embedding is False else [self.word2idx_emb[target_text]]
        return input_ids, attention_mask, target, sentence_id, ws_label, word_indices


def get_model_input(data, device=None, ws=False, use_embedding=False):
    """

    :param data: input_ids1, input_ids2, label_starts, label_ends, true_label
    :return:
    """

    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])
    max_len_words = 0 if use_embedding==False else max([len(x[-1]) for x in data])

    input_ids_list = []
    attention_mask_list = []
    target = []
    sentence_id = []
    ws_label = []
    word_indices = []


    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
        if ws:
            ws_label.append(pad(d[-2], max_len, 0))
        if use_embedding:
            word_indices.append(pad(d[-1], max_len_words, 0))

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(attention_mask_list, dtype=np.int64)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)

    if ws:
        ws_label = np.array(ws_label, dtype=np.int64)
        ws_label = torch.from_numpy(ws_label).to(device)
    if use_embedding:
        word_indices = np.array(word_indices, dtype=np.int64)
        word_indices = torch.from_numpy(word_indices).to(device)
    return input_ids, attention_mask, target, sentence_id, ws_label, word_indices


def test(model, val_loader, file_out=None):
    # model.eval()
    model.train()
    predicts = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id, ws_label, word_indices = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True, ws_label=ws_label, word_indices=word_indices)
            H_attention_mask = torch.triu(
                torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
            H_pred = torch.where(
                output["logits"] >= 0,
                torch.ones(output["logits"].shape).to(device),
                torch.zeros(output["logits"].shape).to(device)
            ) * H_attention_mask

            predict_idx = torch.nonzero(H_pred)
            pred = [[] for i in range(len(H_attention_mask))]
            for idx in predict_idx:
                if idx[2] < target[idx[0]][0]:
                    pred[idx[0]].append([sentence_id[idx[0]], idx[1].item() - 1, idx[2].item() - 1])
                elif idx[1] > target[idx[0]][1]:
                    pred[idx[0]].append([sentence_id[idx[0]], idx[1].item() - 3, idx[2].item() - 3])
            for idx in pred:
                predicts += idx
            pass
    data_json = json.dumps(predicts, indent=1, ensure_ascii=False)
    # 需要修改
    with open(file_out, 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    seed = 77
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)
    # args.use_ws = True
    # args.use_embedding = True
    test_dataset = Dataset("./dataset/cfn_ws/cfn-test-A-ws.json",
                            "./dataset/frame_info.json",
                            tokenizer,
                            ws=args.use_ws,
                            use_embedding=args.use_embedding,
                            embed_file="./dataset/cc.zh.300.cfn.vec")
    file_out = 'dataset/seed77/A_task2_test.json'

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = 1
    config.max_cls = test_dataset.max_cls
    config.use_ws = args.use_ws
    config.use_embedding = args.use_embedding
    if args.use_embedding:
        config.pretrained_embeddings = torch.tensor(test_dataset.embeddings, dtype=torch.float32).to(device)
    model = Model(config)
    ###### 输出model的config和args ######
    print(f'Arguments (args): ')
    print(json.dumps(args.__dict__, indent=2))   
    # print(f"\n Model Config: ")
    print(json.dumps(model.config.__dict__, indent=2)) 
    # load_pretrained_bert(model, args.init_checkpoint)
    # state = torch.load("saves/model_task2_best.bin", map_location='cpu')
    state = torch.load("saves/model_task2_best_robert_large.bin", map_location='cpu')
    # state = torch.load("saves/model_task2_best_roberta_large_targetembedding_ws.bin", map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)


    test_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=test_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device, ws=args.use_ws, use_embedding=args.use_embedding),
        drop_last=False
    )
    test(model, test_loader, file_out)
