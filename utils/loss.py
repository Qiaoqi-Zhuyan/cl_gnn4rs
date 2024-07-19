import torch
import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F

# class BRPLoss(nn.Module):
#     '''
#     Bayesian Personalized Ranking Loss
#     '''
#     def __init__(self, gamma=1e-10):
#         super(BRPLoss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, pos_score, neg_score):
#         loss = -torch.log(self.gamma + sigmoid(pos_score - neg_score)).mean()
#
#         return loss

# class l2_loss(nn.Module):
#     '''
#     l2 norm
#     '''
#     def __init__(self):
#         super(l2_loss, self).__init__()
#
#     def forward(self, *params):
#         reg_loss = 0.0
#         for w in params:
#             reg_loss += torch.sum(torch.pow(w, 2))
#
#         return 0.5*reg_loss
#

# class BPR_loss(nn.Module):
#     def __init__(self, gamma=1e-10):
#         super(BPR_loss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, pos_score, neg_score):
#
#         return -torch.sum(F.logsigmoid(pos_score - neg_score))




# class InfoNCE_loss(nn.Module):
#     def __init__(self):
#         super(InfoNCE_loss, self).__init__()
#
#     def forward(self, pos_score, neg_score, tao):
#         return torch.logsumexp((neg_score - pos_score[:, None]) / tao, dim=1)


def l2_loss(*params):
    reg_loss = 0.
    for w in params:
        reg_loss += torch.sum(torch.pow(w, 2))
    return  0.5 * reg_loss


def bpr_loss(pos_score, neg_score):
    return -torch.sum(F.logsigmoid(pos_score - neg_score))


def infonce_loss(pos_score, total, tao):
    return torch.logsumexp((total - pos_score[:, None]) / tao, dim=1)

