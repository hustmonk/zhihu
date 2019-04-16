import torch
import torch.nn as nn
import logging
from shell import layers
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ReaderNet(nn.Module):
    def __init__(self, args):
        super(ReaderNet, self).__init__()

        self.args = args

        #first layer
        self.linear = nn.Linear(args.embedding_dim, 250)

    def forward(self, inputs):
        passage, passage_pooled, passage_mask, answer1, answer1_pooled, answer1_mask, answer2, answer2_pooled, answer2_mask = inputs

        passage_pooled = self.linear(passage_pooled)
        answer1_pooled = self.linear(answer1_pooled).unsqueeze(1)
        answer2_pooled = self.linear(answer2_pooled).unsqueeze(1)

        #first layer
        score1 = answer1_pooled.bmm(passage_pooled.unsqueeze(2)).squeeze(2)
        score2 = answer2_pooled.bmm(passage_pooled.unsqueeze(2)).squeeze(2)

        answer = torch.cat([score1, score2], 1)
        answer = F.log_softmax(answer, dim=-1)

        return answer

