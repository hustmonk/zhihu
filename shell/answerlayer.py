import torch
import torch.nn as nn
import logging
from shell import layers
import torch.nn.functional as F
logger = logging.getLogger(__name__)

class AnswerLayer(nn.Module):
    def __init__(self, args):
        super(AnswerLayer, self).__init__()
        self.args = args
        #answer info
        self.answer_match1 = layers.SeqAttnMatch(args.embedding_dim)
        self.answer_match2 = layers.SeqAttnMatch(args.hidden_size * 2)
        self.answer_match3 = layers.SeqAttnMatch(args.hidden_size * 2)

        self.answer_lstm1 = layers.StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size)
        self.answer_lstm2 = layers.StackedBRNN(input_size=args.hidden_size * 2 * 2 + args.embedding_dim, hidden_size=args.hidden_size)
        self.answer_lstm3 = layers.StackedBRNN(input_size=args.hidden_size * 2 * 3 + args.embedding_dim, hidden_size=args.hidden_size)
        self.scorer1 = layers.ScoreLayer(args.hidden_size * 2)
        self.scorer2 = layers.ScoreLayer(args.hidden_size * 2)
        self.scorer3 = layers.ScoreLayer(args.hidden_size * 2)

    def forward(self, passageinfo, passage_mask, answer, answer_mask):
        passage, passage1, passage2, passage3 = passageinfo

        match = self.answer_match1(answer, passage, passage_mask)
        weight = 0.3
        answer0 = match * weight + answer * (1 - weight)
        answer1 = self.answer_lstm1(answer0, answer_mask)
        score1 = self.scorer1(passage1, passage_mask, answer1, answer_mask)

        match = self.answer_match2(answer1, passage1, passage_mask)
        answer2 = self.answer_lstm2(torch.cat([answer, answer1, match], -1), answer_mask)
        score2 = self.scorer2(passage2, passage_mask, answer2, answer_mask)

        match = self.answer_match3(answer2, passage2, passage_mask)
        answer3 = self.answer_lstm3(torch.cat([answer, answer1, answer2, match], -1), answer_mask)
        score3 = self.scorer3(passage3, passage_mask, answer3, answer_mask)
        if self.args.score_type == 0:
            return score1 + score2 + score3
        elif self.args.score_type == 1:
            return score1
        elif self.args.score_type == 2:
            return score2
        elif self.args.score_type == 3:
            return score3