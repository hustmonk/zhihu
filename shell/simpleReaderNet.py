import torch
import torch.nn as nn
import logging
from shell import layers
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

logger = logging.getLogger(__name__)

class ReaderNet(BertPreTrainedModel):
    def __init__(self, config):
        super(ReaderNet, self).__init__(config)

        self.bert = BertModel(config)
        self.scorer = layers.ScoreLayer(config.hidden_size)
        self.apply(self.init_bert_weights)

    def bertfeature(self, inputs):
        outputs = []
        for (i) in range(0, len(inputs), 3):
            ids = inputs[i]
            segments = inputs[i + 1]
            mask = inputs[i + 2]
            encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
            mask = 1 - mask
            embedding = F.dropout(encoded_layers[-1], p=0.2, training=self.training)
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

