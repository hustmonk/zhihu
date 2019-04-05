import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

PROJECTION_SIZE = 250 #attention相关的
PROJECTION_DROPOUT = 0.1
class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers = 1,
                 dropout_rate=0.2, dropout_output=True, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        self.linear = nn.Linear(input_size, hidden_size * 3)
        input_size = 3 * hidden_size

        for i in range(num_layers):
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))
            input_size = 2 * hidden_size

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if self.linear != None:
            x = F.relu(self.linear(x))
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            self.rnns[i].flatten_parameters()
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            self.rnns[i].flatten_parameters()
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class SeqAttnMatch(nn.Module):
    def __init__(self, input_size):
        super(SeqAttnMatch, self).__init__()
        self.match = SeqAttnMatchKeyValue(input_size)

    def forward(self, x, y_key, y_mask):
        return self.match(x, y_key, y_key, y_mask)

class SeqAttnMatchKeyValue(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size):
        super(SeqAttnMatchKeyValue, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        if self.output_size > PROJECTION_SIZE:
            self.output_size = PROJECTION_SIZE
        self.linear = nn.Linear(input_size, self.output_size)

    def forward(self, x, y_key, y_value, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """

        x_proj = self.linear(x)
        x_proj = F.relu(x_proj)
        y_proj = self.linear(y_key)
        y_proj = F.relu(y_proj)
        x_proj = F.dropout(x_proj, p=PROJECTION_DROPOUT, training=self.training)
        y_proj = F.dropout(y_proj, p=PROJECTION_DROPOUT, training=self.training)

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())

        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y_key.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y_key.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y_value)
        return matched_seq

class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear1 = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear1(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)

class LinearScore(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, input_size):
        super(LinearScore, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask = None):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.linear(x).view(x.size(0), -1)
        if x_mask is not None:
            scores.data.masked_fill_(x_mask.data, -float('inf'))

        return scores

class ScoreLayer(nn.Module):

    def __init__(self, input_size):
        super(ScoreLayer, self).__init__()
        self.answer_self_attn = LinearSeqAttn(input_size)
        self.passage_self_attn = LinearSeqAttn(input_size)
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, passage, passage_mask, answer, answer_mask):
        answer = self.answer_self_attn(answer, answer_mask).unsqueeze(1)
        passage = self.passage_self_attn(passage, passage_mask)
        answer = self.linear(answer)
        passage = self.linear(passage)

        answer = answer.bmm(passage.unsqueeze(2)).squeeze(2)
        return answer