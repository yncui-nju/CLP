from src.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatELoss(nn.Module):
    def __init__(self, args, kg, model):
        super(RotatELoss, self).__init__()
        self.args = args
        self.kg = kg
        self.model = model

    def forward(self, input, target, subsampling_weight=None, confidence=1.0, neg_ratio=10):
        p_score, n_score = self.split_pn_score(input, target, neg_ratio)
        p_score = F.logsigmoid(p_score) * confidence
        p_loss = -(subsampling_weight * p_score).sum(dim=-1)/subsampling_weight.sum()
        if neg_ratio > 0:
            # instance-level
            n_score = (F.softmax(n_score, dim=-1).detach()
                              * F.logsigmoid(-n_score)).sum(dim=-1) * confidence
            n_loss = -(subsampling_weight * n_score).sum(dim=-1) / subsampling_weight.sum()
            loss = (p_loss + n_loss)/2
        else:
            # schema-level
            loss = p_loss
        return loss


    def split_pn_score(self, score, label, neg_ratio):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return:
        '''
        if neg_ratio <= 0:
            return score, 0
        p_score = score[torch.where(label > 0)]
        n_score = score[torch.where(label < 0)].reshape(-1, neg_ratio)
        return p_score, n_score



