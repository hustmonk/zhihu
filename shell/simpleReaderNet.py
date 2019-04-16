import torch
import torch.nn as nn
import logging
from shell import layers
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

logger = logging.getLogger(__name__)

class ReaderNet(nn.Module):
    def __init__(self, args):
        super(ReaderNet, self).__init__()

        self.bert = BertModel.from_pretrained(args.bert_base_uncased)
        self.linear = nn.Linear(args.embedding_dim, 250)

        self.scorer = layers.ScoreLayer(250)

    def bertfeature(self, inputs):
        outputs = []
        for (i) in range(0, len(inputs), 3):
            ids = inputs[i]
            segments = inputs[i + 1]
            mask = inputs[i + 2]
            encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
            mask = 1 - mask
            embedding = F.dropout(encoded_layers[-1], p=0.4, training=self.training)
            embedding = F.relu(self.linear(embedding))
            embedding = F.dropout(embedding, p=0.4, training=self.training)

            outputs = outputs + [embedding, mask]
        return outputs

    def forward(self, inputs):
        inputs = self.bertfeature(inputs)
        passage, passage_mask, answer1, answer1_mask, answer2, answer2_mask = inputs

        #first layer
        score1 = self.scorer(passage, passage_mask, answer1, answer1_mask)
        score2 = self.scorer(passage, passage_mask, answer2, answer2_mask)

        answer = torch.cat([score1, score2], 1)
        answer = F.log_softmax(answer, dim=-1)

        return answer

