import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo
import copy, logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MTLSTM(nn.Module):

    def __init__(self, args):
        """Initialize an MTLSTM.

        Arguments:
            n_vocab (bool): If not None, initialize MTLSTM with an embedding matrix with n_vocab vectors
            vectors (Float Tensor): If not None, initialize embedding matrix with specified vectors
            residual_embedding (bool): If True, concatenate the input embeddings with MTLSTM outputs during forward
        """
        super(MTLSTM, self).__init__()
        self.args = args
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True, batch_first=True)
        for p in self.parameters():
            p.requires_grad = False

    def rnnforward(self, rnn, inputs, x_mask, hidden):
        """
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        lens, indices = torch.sort(lengths, 0, True)
        outputs, hidden_t = rnn(pack(inputs[indices], lens.tolist(), batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        return outputs
        """
        outputs, hidden_t = rnn(inputs, hidden)
        return outputs
    def forward(self, inputs, x_mask, hidden=None):
        outputs1 = self.rnnforward(self.rnn1, inputs, x_mask, hidden)
        outputs2 = self.rnnforward(self.rnn2, outputs1, x_mask, hidden)
        return [outputs1, outputs2]

    def load_cove(self, cove_file):
        logger.info("Load cove [%s]" % (cove_file))
        state_dict = torch.load(cove_file, map_location=lambda storage, loc: storage)
        state_dict_1 = self.rnn1.state_dict()
        for (pname, pstate) in state_dict_1.items():
            state_dict_1[pname] = state_dict[pname]
        self.rnn1.load_state_dict(state_dict_1)

        state_dict_2 = self.rnn2.state_dict()
        for (pname, pstate) in state_dict_2.items():
            newname = pname.replace('0', '1')
            state_dict_2[pname] = state_dict[newname]
        self.rnn2.load_state_dict(state_dict_2)

if __name__ == "__main__":
    mtlstm = MTLSTM()
    for (name, p) in mtlstm.named_parameters():
        print(name, p.requires_grad)
    mtlstm.load_cove("data/embeddings/wmtlstm-b142a7f2.pth")
