import torch
import torch.nn as nn
from shell import mlstm
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class PretrainedDataLayers(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(PretrainedDataLayers, self).__init__()
        self.args = args
        self.args.embedding_dim = self.args.glove_embedding_dim
        if self.args.use_cove:
            self.args.embedding_dim += 600

    def load_pretrained_dict(self, word_dict, emb=None):
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(len(word_dict),
                                      self.args.glove_embedding_dim,
                                      padding_idx=0)

        self.load_embeddings(word_dict, self.args.embedding_file)
        if self.args.use_cove:
            self.mlstm = mlstm.MTLSTM(self.args)
            self.mlstm.load_cove(self.args.cove_file)
        for p in self.parameters():
            p.requires_grad = False

    def embed_dropout(self, x, x_mask):
        x = self.embedding(x)
        if self.args.use_cove:
            cove = self.mlstm(x, x_mask)
            x = torch.cat([x, cove[-1]], -1)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        return x

    def forward(self, inputs):
        passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask, \
        answer1, answer1_mask, answer2, answer2_mask, qanswer1, qanswer1_mask, qanswer2, qanswer2_mask = inputs

        passage = self.embed_dropout(passage, passage_mask)
        question = self.embed_dropout(question, question_mask)
        questioninfo = self.embed_dropout(questioninfo, questioninfo_mask)
        answer1 = self.embed_dropout(answer1, answer1_mask)
        qanswer1 = self.embed_dropout(qanswer1, qanswer1_mask)
        answer2 = self.embed_dropout(answer2, answer2_mask)
        qanswer2 = self.embed_dropout(qanswer2, qanswer2_mask)

        return [passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask,
         answer1, answer1_mask, answer2, answer2_mask, qanswer1, qanswer1_mask, qanswer2, qanswer2_mask]

    def load_embeddings(self, word_dict, embedding_file):
        embedding = self.embedding.weight.data
        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        unk_vec = torch.Tensor([0 for i in range(embedding.size(1))])
        unk_count = 0
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(word_dict), embedding_file))
        f = open(embedding_file, encoding='utf-8')

        for line in f:
            w = word_dict.normalize(line[:40].split(' ')[0])
            if w in word_dict:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                unk_vec.add_(vec)
                unk_count = unk_count + 1
                if w not in vec_counts:
                    vec_counts[w] = 1
                    embedding[word_dict[w]].copy_(vec)
                else:
                    logging.warning(
                        'WARN: Duplicate embedding found for %s' % w
                    )
                    vec_counts[w] = vec_counts[w] + 1
                    embedding[word_dict[w]].add_(vec)

        f.close()
        for w, c in vec_counts.items():
            embedding[word_dict[w]].div_(c)
        embedding[word_dict["<UNK>"]].copy_(unk_vec.div_(unk_count))

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(word_dict)))
