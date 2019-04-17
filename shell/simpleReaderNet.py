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

        self.bert = BertModel.from_pretrained(args.bert_model)
        self.linear = nn.Linear(args.embedding_dim, 1)

    def forward(self, inputs):
        scores = []
        for (i) in range(0, len(inputs), 3):
            ids = inputs[i]
            segments = inputs[i + 1]
            mask = inputs[i + 2]
            encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
            encoder = encoded_layers[-1][:, 0, :]
            encoder = F.dropout(encoder, p=0.4, training=self.training)
            score = self.linear(encoder)
            scores.append(score)
        return F.log_softmax(torch.cat(scores, 1), dim=-1)

