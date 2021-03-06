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

class ReaderNet(nn.Module):
    def __init__(self, args):
        super(ReaderNet, self).__init__()

        self.args = args

        #first layer
        self.question_match1 = layers.SeqAttnMatch(args.embedding_dim)
        self.questioninfo_match1 = layers.SeqAttnMatch(args.embedding_dim)
        self.passage_lstm1 = layers.StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size)
        self.question_lstm1 = layers.StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size)
        self.questioninfo_lstm1 = layers.StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size)

        #second layer
        self.question_match2 = layers.SeqAttnMatch(args.hidden_size * 2)
        self.questioninfo_match2 = layers.SeqAttnMatch(args.hidden_size * 2)
        passage_input_size = args.hidden_size * 2 * 3 + args.embedding_dim
        question_input_size = args.hidden_size * 2 + args.embedding_dim
        self.passage_lstm2 = layers.StackedBRNN(input_size=passage_input_size, hidden_size=args.hidden_size)
        self.question_lstm2 = layers.StackedBRNN(input_size=question_input_size, hidden_size=args.hidden_size)
        self.questioninfo_lstm2 = layers.StackedBRNN(input_size=question_input_size, hidden_size=args.hidden_size)

        #third layer
        self.question_match3 = layers.SeqAttnMatchKeyValue(args.hidden_size * 2 * 2)
        self.questioninfo_match3 = layers.SeqAttnMatchKeyValue(args.hidden_size * 2 * 2)
        passage_input_size = args.hidden_size * 2 * 4 + args.embedding_dim
        self.passage_lstm3 = layers.StackedBRNN(input_size=passage_input_size, hidden_size=args.hidden_size)
        self.answer = AnswerLayer(args)
        self.qanswer = AnswerLayer(args)

    def forward(self, inputs):
        passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask, \
            answer1, answer1_mask, answer2, answer2_mask, qanswer1, qanswer1_mask, qanswer2, qanswer2_mask = inputs

        #first layer
        match1 = self.question_match1(passage, question, question_mask)
        match2 = self.questioninfo_match1(passage, questioninfo, questioninfo_mask)
        passage0 = passage + match1 + match2
        passage1 = self.passage_lstm1(passage0, passage_mask)
        question1 = self.question_lstm1(question, question_mask)
        questioninfo1 = self.questioninfo_lstm1(questioninfo, questioninfo_mask)

        #second layer
        match1 = self.question_match2(passage1, question1, question_mask)
        match2 = self.questioninfo_match2(passage1, questioninfo1, questioninfo_mask)
        passage2 = self.passage_lstm2(torch.cat([passage, match1, match2, passage1], -1), passage_mask)
        question2 = self.question_lstm2(torch.cat([question, question1], -1), question_mask)
        questioninfo2 = self.questioninfo_lstm2(torch.cat([questioninfo, questioninfo1], -1), questioninfo_mask)

        #third layer
        pm = torch.cat([passage1, passage2], -1)
        qm = torch.cat([question1, question2], -1)
        qim = torch.cat([questioninfo1, questioninfo2], -1)
        match1 = self.question_match3(pm, qm, question2, question_mask)
        match2 = self.questioninfo_match3(pm, qim, questioninfo2, questioninfo_mask)
        passage3 = self.passage_lstm3(torch.cat([passage, passage1, passage2, match1, match2], -1), passage_mask)

        #finnaly passage information
        passageinfo = [passage, passage1, passage2, passage3]

        #answer info
        answer1 = self.answer(passageinfo, passage_mask, answer1, answer1_mask)
        answer2 = self.answer(passageinfo, passage_mask, answer2, answer2_mask)
        qanswer1 = self.qanswer(passageinfo, passage_mask, qanswer1, qanswer1_mask)
        qanswer2 = self.qanswer(passageinfo, passage_mask, qanswer2, qanswer2_mask)

        answer = torch.cat([answer1 + qanswer1, answer2 + qanswer2], 1)
        answer = F.log_softmax(answer, dim=-1)

        return answer

