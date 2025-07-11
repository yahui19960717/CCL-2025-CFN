#!/usr/bin/python3

import os
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from dataset_task1 import Dataset
from params1 import args
from model_task1 import Model
import json
from datetime import datetime
from differance_time import get_time_diff
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


def get_model_input(data, device=None, ws=False, use_embedding=False):
    """

    :param data: input_ids, attention_mask, target, label, sentence_id, pos, dep, ws_label, word_embedding
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
    labels = []
    sentence_id = []
    pos = []
    dep = []
    ws_label = []
    word_indices = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])
        # 使用条件表达式来处理 pos 和 dep，避免重复的判断
        pos.append(d[5] if d[5] is not None else None)
        dep.append(d[6] if d[6] is not None else None)
        if ws:
            ws_label.append(pad(d[-2], max_len, 0))
        if use_embedding:
            word_indices.append(pad(d[-1], max_len_words, 0))

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(attention_mask_list, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)

    input_ids = torch.from_numpy(input_ids).to(device)
    # import pdb;pdb.set_trace()
    attention_mask = torch.from_numpy(attention_mask).to(device)
    labels = torch.from_numpy(labels).to(device)

    if ws:
        ws_label = np.array(ws_label, dtype=np.int64)
        ws_label = torch.from_numpy(ws_label).to(device)
    if use_embedding:
        word_indices = np.array(word_indices, dtype=np.int64)
        word_indices = torch.from_numpy(word_indices).to(device)
    return input_ids, attention_mask, target, labels, sentence_id, pos, dep, ws_label, word_indices


def eval(model, val_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, labels, sentence_id, pos, dep, ws_label, word_indices = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device, for_test=True, pos=pos, dep=dep, ws_label=ws_label, word_indices=word_indices)
            logits = output["logits"]
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            correct += (pred == labels).sum().item() 
            total += len(pred)
            
    return correct / (total + 1e-6)


def train(model, train_loader, val_loader, exp_name):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # *******************  NoisyTune  ****************

    noise_lambda = 0.15
    for name, para in param_optimizer:
        model.state_dict()[name][:] += \
            (torch.rand(para.size()).to(device) - 0.5) * \
            noise_lambda * torch.std(para)

    # ***********************************************


    total_steps = int(len(train_loader) * args.num_train_epochs /
                      args.accumulate_gradients)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr)

    # adv = FGM(bert_model) if args.with_adv_train else None
    global_step = 0
    best_acc = 0.0
    fgm = FGM(model)
    for i_epoch in range(1, 1 + args.num_train_epochs):
        total_loss = 0.0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            global_step += 1

            input_ids, attention_mask, target, labels, sentence_id, pos, dep, ws_label, word_indices = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device, pos=pos, dep=dep, ws_label=ws_label, word_indices=word_indices)

            loss = output['loss']

            total_loss += loss.item()

            if (step + 1) % 1000 == 0:
                print(
                    f'loss: {total_loss / (step + 1)}')

            loss.backward()

            # fgm.attack()  # embedding被修改了
            # # optimizer.zero_grad()  # 如果不想累加梯度，就把这里的注释取消
            # loss_sum = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=labels, device=device)['loss']
            # loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复Embedding的参数

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if (step + 1) % args.accumulate_gradients == 0:
                lr_this_step = args.lr * \
                               warmup_linear(global_step / total_steps,
                                             args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # break

        acc = eval(model, val_loader)

        if acc > best_acc:
            print(f'saved! new best acc {acc}, ori_acc {best_acc}')
            best_acc = acc
            model_to_save = model.module if hasattr(model, 'module') else model
            os.makedirs('saves', exist_ok=True)
            torch.save(model_to_save.state_dict(), exp_name)
        else:
            print(f'current acc: {acc}')

        # train_loader.dataset.gen_data()


def load_pretrained_bert(bert_model, init_checkpoint):
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location='cpu')
        if 'model_bert_best' in init_checkpoint:
            bert_model.load_state_dict(state['model_bert'], strict=False)
        else:
            state = {k.replace('bert.', '').replace('roformer.', ''): v for k, v in state.items() if
                     not k.startswith('cls.')}
            # state['embeddings.token_type_embeddings.weight'] = state['embeddings.token_type_embeddings.weight'][:2, :]
            bert_model.load_state_dict(state, strict=False)


if __name__ == '__main__':
    start_time = datetime.now()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    seed=1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    # # 是否使用ws
    args.use_ws = True
    # # 是否使用word embedding
    args.use_embedding = True
    train_dataset = Dataset("./dataset/cfn_ws/cfn-train-ws.json",
                            "./dataset/frame_info.json",
                            tokenizer,
                            ws=args.use_ws,
                            use_embedding=args.use_embedding,
                            embed_file="./dataset/cc.zh.300.cfn.vec") 
    dev_dataset = Dataset("./dataset/cfn_ws/cfn-dev-ws.json",
                          "./dataset/frame_info.json",
                          tokenizer,
                          ws=args.use_ws,
                          use_embedding=args.use_embedding,
                          embed_file="./dataset/cc.zh.300.cfn.vec")
    exp_name = f'saves/model_task1_best_roberta_large_word_embedding_ws_falsefreeze.bin' # 使用roberta-large，不使用ws，不冻结词embedding
    # 是否使用focal loss   
    # args.focal_loss = True
    # args.gamma = 4.0
    # 是否使用circle loss
    # args.circle_loss = True
    # 是否使用词性
    # args.use_pos = True
    # # 是否使用依存关系
    # args.use_dep = True



    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = train_dataset.num_labels
    config.use_ws = args.use_ws
    config.use_embedding = args.use_embedding
    config.embed_file = args.embed_file
    config.focal_loss = args.focal_loss
    config.gamma = args.gamma
    config.use_pos = args.use_pos
    config.use_dep = args.use_dep
    if args.use_embedding:
        config.pretrained_embeddings = torch.tensor(train_dataset.embeddings, dtype=torch.float32).to(device)
    model = Model(config)

    ###### 输出model的config和args ######
    print(f'Arguments (args): ')
    print(json.dumps(args.__dict__, indent=2))
    # print(f"\n Model Config: ")
    # print(json.dumps(model.config.__dict__, indent=2))
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)

    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device, ws=args.use_ws, use_embedding=args.use_embedding),
        drop_last=True
    )

    val_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=dev_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device, ws=args.use_ws, use_embedding=args.use_embedding),
        drop_last=False
    )
    train(model, train_loader, val_loader, exp_name)
    end_time = datetime.now()
    get_time_diff(start_time, end_time)
