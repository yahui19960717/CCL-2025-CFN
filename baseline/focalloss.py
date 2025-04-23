import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        input_soft = F.softmax(input, dim=1)
        log_input_soft = F.log_softmax(input, dim=1)
        loss = - target * (1 - input_soft) ** self.gamma * log_input_soft

        if self.alpha is not None:
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = alpha_weight * loss

        if self.reduction == 'mean':
            return torch.mean(torch.sum(loss, dim=-1))
        elif self.reduction == 'sum':
            return torch.sum(torch.sum(loss, dim=-1))
        else:
            return loss


# class FocalLossWithLogits(nn.Module):
#     def __init__(self, gamma=2, alpha=None, reduction='mean'):
#         super(FocalLossWithLogits, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.reduction = reduction

#     def forward(self, input, target):
#         input_soft = torch.sigmoid(input) if input.shape[1] == 1 else F.softmax(input, dim=1)
#         log_input_soft = F.log_sigmoid(input) if input.shape[1] == 1 else F.log_softmax(input, dim=1)
#         loss = - target * (1 - input_soft) ** self.gamma * log_input_soft

#         if self.alpha is not None:
#             alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
#             loss = alpha_weight * loss

#         if self.reduction == 'mean':
#             return torch.mean(torch.sum(loss, dim=-1))
#         elif self.reduction == 'sum':
#             return torch.sum(torch.sum(loss, dim=-1))
#         else:
#             return loss


class FocalLossWithLogits(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha   # 平衡因子
        self.gamma = gamma   # 焦点因子
        self.reduction = reduction  # 计算损失的方式（'mean', 'sum', 'none'）

    def forward(self, input, target):
        # input: shape [batch_size, num_classes] (logits)
        # target: shape [batch_size], 真实标签是整数索引
        # 对 logits 应用 softmax 计算每个类别的概率分布
        input_softmax = F.softmax(input, dim=1)  # 计算每个类别的概率分布
        log_input_softmax = F.log_softmax(input, dim=1)  # 计算对数概率

        # 使用 gather 从 log_input_softmax 中选择正确类别的对数概率
        p_t = input_softmax.gather(dim=1, index=target.unsqueeze(1))  # shape [batch_size, 1]
        log_p_t = log_input_softmax.gather(dim=1, index=target.unsqueeze(1))  # shape [batch_size, 1]

        # 焦点因子 (1 - p_t) ** gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal Loss 公式
        if self.alpha is not None:
            loss = -self.alpha * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t

        # 如果需要平均或者求和
        if self.reduction == 'mean':
            return loss.mean()  # 计算平均损失
        elif self.reduction == 'sum':
            return loss.sum()  # 计算总损失
        else:
            return loss  # 返回逐个样本的损失
        # input_softmax = F.softmax(input, dim=1)
        # log_input_softmax = F.log_softmax(input, dim=1)
        # p_t = input_softmax.gather(dim=1, index=target.unsqueeze(1))
        # log_p_t = log_input_softmax.gather(dim=1, index=target.unsqueeze(1))
        # focal_weight = (1 - p_t) ** self.gamma
        # loss = -focal_weight * log_p_t  # 如果 alpha 为 None，则不乘以 alpha

        # if self.alpha is not None:
        #     alpha_weight = self.alpha.gather(dim=0, index=target) # 需要调整 alpha 的形状
        #     loss = alpha_weight.unsqueeze(1) * loss

        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:
        #     return loss