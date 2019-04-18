import copy
import json
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from .simpleReaderNet import ReaderNet
import numpy
from torch.autograd import Variable
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logger = logging.getLogger(__name__)

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

        #set optimizer
        param_optimizer = list(self.network.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        def ignore(n):
            ignores = ['pooler', 'bert.embeddings', 'bert.encoder.layer.0.', 'bert.encoder.layer.1.', 'bert.encoder.layer.2.',
                   'bert.encoder.layer.3.', 'bert.encoder.layer.4.', 'bert.encoder.layer.5.', 'bert.encoder.layer.6.',
                       'bert.encoder.layer.7.', 'bert.encoder.layer.8.'
                   ]
            for k in ignores:
                if k in n[0]:
                    return True
            return False
        param_optimizer = [n for n in param_optimizer if ignore(n) == False]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = 10000/args.batch_size * 10
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

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

