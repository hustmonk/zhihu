import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class PretrainedDataLayers(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(PretrainedDataLayers, self).__init__()
        self.model_args = args

    def load_pretrained_dict(self, args, word_dict, emb=None):

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(len(word_dict),
                                      self.model_args.embedding_dim,
                                      padding_idx=0)

        self.load_embeddings(word_dict, args.embedding_file)

        for p in self.parameters():
            p.requires_grad = False

    def embed_dropout(self, x):
        x = self.embedding(x)
        x = nn.functional.dropout(x, p=0.3, training=self.training)
        return x

    def forward(self, inputs):
        [passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask,
         answer1, answer1_mask, answer2, answer2_mask] = inputs

        passage = self.embed_dropout(passage)
        question = self.embed_dropout(question)
        questioninfo = self.embed_dropout(questioninfo)
        answer1 = self.embed_dropout(answer1)
        answer2 = self.embed_dropout(answer2)

        return [passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask,
         answer1, answer1_mask, answer2, answer2_mask]

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