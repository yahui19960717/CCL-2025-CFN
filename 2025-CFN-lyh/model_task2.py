import torch
import torch.nn as nn
from transformers import BertModel
from affine import Biaffine

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.inner_dim = 64
        self.bert = BertModel(config)
        # self.lstm = nn.LSTM(768, 768 // 2, num_layers=1, batch_first=True,
        #                     bidirectional=True)
    
        input_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.config.use_ws:
            self.ws_embedding = nn.Embedding(5, 16)
            input_size += 16
        if self.config.use_embedding:
            self.word_embedding = nn.Embedding.from_pretrained(self.config.pretrained_embeddings, freeze=True)
            input_size += 300
            
        self.dense = nn.Linear(input_size, config.num_labels * self.inner_dim * 2)
        # self.biaffine = Biaffine(self.inner_dim, config.num_labels, bias_x=True, bias_y=True)
        # self.width_embedding = nn.Embedding(25, 1) # 宽度 embedding 维度设为 1

         

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, target=None, device=None, for_test=False, ws_label=None, word_indices=None):
        # labels: [batch_size, seq_len, seq_len]
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # hidden_token: [batch_size, seq_len, hidden_size]
        hidden_token = bert_out.last_hidden_state
        # hidden_token = (bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 2
        if self.config.use_ws:
            ws_emb = self.ws_embedding(ws_label)
            # hidden_token = hidden_token + ws_emb
            hidden_token = torch.cat([hidden_token, ws_emb], dim=-1)
        if self.config.use_embedding:
            word_emb = self.word_embedding(word_indices)
            word_emb = word_emb.expand(-1, hidden_token.shape[1], -1)
            hidden_token = torch.cat([hidden_token, word_emb], dim=-1)
        outputs = self.dense(hidden_token)

        # lstm_out, (hidden, _) = self.lstm(hidden_token)
        # outputs = self.dense(lstm_out)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)ent_type_size=1
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim) 起始和结束向量的表示
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # pos_emb:(batch_size, seq_len, inner_dim) 生成位置编码
        pos_emb = self.sinusoidal_position_embedding(hidden_token.shape[0], hidden_token.shape[1], 64, device)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        
        qw2 = qw2.reshape(qw.shape)
        # qw, kw torch.Size([batch_size, seq_len, label_type, self.inner_dim])
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        
        # 它不是像传统方法那样预测每个 token 的标签，而是学习预测哪些位置可以作为实体的开始和结束。
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        # logits = self.biaffine(qw, kw)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = logits / self.inner_dim ** 0.5
        logits = logits.squeeze(1)
        if for_test:
            loss = None
        else:
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(H_logits.view(-1, 2), labels.view(-1))
            loss = self.compute_loss(logits, labels, attention_mask)
        return {
            "logits": logits,
            "loss": loss
        }

    def compute_loss(self, logits, labels, attention_mask):
        H_attention_mask = torch.triu(
            torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
        loss1 = torch.sum(torch.exp(-logits) * H_attention_mask * labels, dim=(1, 2))
        loss2 = torch.sum(torch.exp(logits) * H_attention_mask * (1 - labels), dim=(1, 2))
        loss = torch.sum(torch.log(1 + loss1 + 1e-9) + torch.log(1 + loss2 + 1e-9)) / (H_attention_mask.shape[0])
        return loss
