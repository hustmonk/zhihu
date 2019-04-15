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
        self.scorer = layers.ScoreLayer(args.embedding_dim)

    def forward(self, inputs):
        passage, passage_mask, answer1, answer1_mask, answer2, answer2_mask = inputs

        #first layer
        score1 = self.scorer(passage, passage_mask, answer1, answer1_mask)
        score2 = self.scorer(passage, passage_mask, answer2, answer2_mask)

        answer = torch.cat([score1, score2], 1)
        answer = F.log_softmax(answer, dim=-1)

        return answer

