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
        self.passage_lstm = layers.StackedBRNN(input_size=300, hidden_size=args.hidden_size)
        self.question_lstm = layers.StackedBRNN(input_size=300, hidden_size=args.hidden_size)
        self.answer_lstm = layers.StackedBRNN(input_size=300, hidden_size=args.hidden_size)
        self.questioninfo_lstm = layers.StackedBRNN(input_size=300, hidden_size=args.hidden_size)

        self.match1 = layers.SeqAttnMatchKeyValue(args.hidden_size * 2)
        self.match2 = layers.SeqAttnMatchKeyValue(args.hidden_size * 2)
        self.match3 = layers.SeqAttnMatchKeyValue(args.hidden_size * 2)
        self.match4 = layers.SeqAttnMatchKeyValue(args.hidden_size * 2)

        self.passage_lstm2 = layers.StackedBRNN(input_size=args.hidden_size * 3 * 2, hidden_size=args.hidden_size)
        self.answer_lstm2 = layers.StackedBRNN(input_size=args.hidden_size * 2 * 2, hidden_size=args.hidden_size)

        self.self_attn1 = layers.LinearSeqAttn(args.hidden_size * 2)
        self.self_attn2 = layers.LinearSeqAttn(args.hidden_size * 2)

        self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size * 2)

    def forward(self, inputs):
        passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask, \
            answer1, answer1_mask, answer2, answer2_mask = inputs

        passage = self.passage_lstm(passage, passage_mask)
        question = self.question_lstm(question, question_mask)
        answer1 = self.answer_lstm(answer1, answer1_mask)
        answer2 = self.answer_lstm(answer2, answer2_mask)
        questioninfo = self.questioninfo_lstm(questioninfo, questioninfo_mask)

        match1 = self.match1(passage, question, question_mask)
        match2 = self.match2(passage, questioninfo, questioninfo_mask)

        passage = self.passage_lstm2(torch.cat([passage, match1, match2], -1), passage_mask)

        match3 = self.match3(answer1, passage, passage_mask)
        match4 = self.match3(answer2, passage, passage_mask)

        answer1 = self.answer_lstm2(torch.cat([answer1, match3], -1), answer1_mask)
        answer2 = self.answer_lstm2(torch.cat([answer2, match4], -1), answer2_mask)

        answer1 = self.self_attn1(answer1, answer1_mask).unsqueeze(1)
        answer2 = self.self_attn1(answer2, answer2_mask).unsqueeze(1)
        passage = self.self_attn2(passage, passage_mask)

        answer = torch.cat([answer1, answer2], 1)

        answer = self.linear(answer)
        answer = answer.bmm(passage.unsqueeze(2)).squeeze(2)
        answer = F.log_softmax(answer, dim=-1)

        return answer