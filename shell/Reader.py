import copy
import json
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from .simpleReaderNet import ReaderNet
import numpy
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

logger = logging.getLogger(__name__)

def build_bertfeature(bert, input_ids, segments_ids, attention_mask, training):
    with torch.no_grad():
        encoded_layers, pooled_output = bert(input_ids, segments_ids, attention_mask=attention_mask)
        attention_mask = 1 - attention_mask
        embedding = F.dropout(encoded_layers[-1], p=0.2, training=training)
        return embedding, attention_mask

def bertfeature(bert, inputs, training=False):
    outputs = []
    for (i) in range(0, len(inputs), 3):
        ids = inputs[i]
        segments = inputs[i + 1]
        mask = inputs[i + 2]
        f, m = build_bertfeature(bert, ids, segments, mask, training)
        outputs = outputs + [f, m]
    return outputs

class Reader(object):
    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, state_dict=None, updates=0):
        # Book-keeping.
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' %
                    json.dumps(vars(args), indent=4, sort_keys=True))
        self.args = args
        self.updates = updates
        self.training = False
        self.use_cuda = False

        self.network = ReaderNet(args)
        if state_dict:
            self.network.load_state_dict(state_dict)

        parameters = [p for p in self.network.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(parameters)

        self.bertmodel = BertModel.from_pretrained(args.bert_base_uncased)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        ids, inputs, targets = ex

        if self.use_cuda:
            inputs = [e if e is None or torch.is_tensor(e) == False else Variable(e.cuda(async=True))
                      for e in inputs]
            targets = Variable(targets.cuda(async=True))

        else:
            inputs = [e if e is None or torch.is_tensor(e) == False else Variable(e) for e in inputs]
            targets = Variable(targets)

        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        inputs = bertfeature(self.bertmodel, inputs, training=True)

        # Run forward
        scores = self.network(inputs)
        loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(scores, targets)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1
        if torch.__version__ > "0.4":
            return loss.item(), len(ids)
        else:
            return loss.data[0], len(ids)

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, inputs):
        if self.use_cuda:
            inputs = [e if e is None or torch.is_tensor(e) == False else Variable(e.cuda(async=True))
                      for e in inputs]
        else:
            inputs = [e if e is None or torch.is_tensor(e) == False else Variable(e) for e in inputs]
        # Eval mode
        self.network.eval()

        # Run forward
        inputs = bertfeature(self.bertmodel, inputs)

        # Run forward
        scores = self.network(inputs)
        return torch.max(scores, -1)[1].data.cpu().numpy().tolist()

    def save(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        args = saved_params['args']
        updates = saved_params["updates"]
        model = Reader(args, state_dict, updates)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()
        self.bertmodel = self.bertmodel.cuda()

