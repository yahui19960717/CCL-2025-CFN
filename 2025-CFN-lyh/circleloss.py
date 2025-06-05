import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, margin=0.25, scale=64.0):
        """
        CircleLoss 构造函数
        :param margin: 阈值 m，控制相似度的下限
        :param scale: 一个缩放因子，控制损失的缩放
        """
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        """
        计算 Circle Loss
        :param embeddings: 形状为 [batch_size, embedding_dim] 的嵌入向量
        :param labels: 形状为 [batch_size] 的标签
        :return: 计算出的 Circle Loss
        """
        # 计算嵌入向量之间的余弦相似度
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # 获取标签是否相同
        labels_exp = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 计算正样本和负样本的相似度
        pos_similarity = similarity_matrix * labels_exp.float()  # 正样本相似度
        neg_similarity = similarity_matrix * (~labels_exp).float()  # 负样本相似度

        # 计算正样本的损失（最大化正样本相似度）
        pos_loss = torch.clamp(self.margin - pos_similarity, min=0)

        # 计算负样本的损失（最小化负样本相似度）
        neg_loss = torch.clamp(neg_similarity - self.margin, min=0)

        # 结合正负样本损失，进行加权求和
        loss = torch.sum(pos_loss + neg_loss) / embeddings.size(0)

        return loss