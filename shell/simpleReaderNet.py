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
        self.linear = nn.Linear(args.embedding_dim * 2, args.hidden_size * 2)

        self.scorer1 = layers.ScoreLayer(args.hidden_size * 2)

    def forward(self, passageinfo, passage_mask, answer, answer_mask):
        passage, passage1 = passageinfo

        match = self.answer_match1(answer, passage, passage_mask)
        answer1 = self.linear(torch.cat([answer, match], -1))
        score1 = self.scorer1(passage1, passage_mask, answer1, answer_mask)
        return score1

class ReaderNet(nn.Module):
    def __init__(self, args):
        super(ReaderNet, self).__init__()

        self.args = args

        #first layer
        self.question_match1 = layers.SeqAttnMatch(args.embedding_dim)
        self.questioninfo_match1 = layers.SeqAttnMatch(args.embedding_dim)
        self.passage_linear = nn.Linear(args.embedding_dim * 3, args.hidden_size * 2)
        self.answer = AnswerLayer(args)
        self.qanswer = AnswerLayer(args)

    def forward(self, inputs):
        passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask, \
            answer1, answer1_mask, answer2, answer2_mask, qanswer1, qanswer1_mask, qanswer2, qanswer2_mask = inputs

        #first layer
        match1 = self.question_match1(passage, question, question_mask)
        match2 = self.questioninfo_match1(passage, questioninfo, questioninfo_mask)
        passage1 = F.relu(self.passage_linear(torch.cat([passage, match1, match2], -1)))

        #second layer

        #finnaly passage information
        passageinfo = [passage, passage1]

        #answer info
        answer1 = self.answer(passageinfo, passage_mask, answer1, answer1_mask)
        answer2 = self.answer(passageinfo, passage_mask, answer2, answer2_mask)
        qanswer1 = self.qanswer(passageinfo, passage_mask, qanswer1, qanswer1_mask)
        qanswer2 = self.qanswer(passageinfo, passage_mask, qanswer2, qanswer2_mask)

        answer = torch.cat([answer1 + qanswer1, answer2 + qanswer2], 1)
        answer = F.log_softmax(answer, dim=-1)

        return answer

