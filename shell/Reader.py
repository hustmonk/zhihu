import copy
import json
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from .ReaderNet import ReaderNet
from .pretraineddatalayers import PretrainedDataLayers
import numpy
from torch.autograd import Variable

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
        self.network = ReaderNet(args)
        self.updates = updates
        self.training = False
        self.use_cuda = False

        if state_dict:
            self.network.load_state_dict(state_dict)

        self.pretraindatalayers = PretrainedDataLayers(args)

        parameters = [p for p in self.network.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(parameters)

    def set_training(self, training = False):
        self.training = training

    def load_pretrained_dict(self, words_dict):
        self.word_dict = words_dict
        self.pretraindatalayers.load_pretrained_dict(self.args, words_dict)
        for idx, m in enumerate(self.network.named_modules()):
            logger.info('NETWORK_GRAPH:\n%s' % str(m))
            break

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
        self.pretraindatalayers.train()
        self.network.train()

        inputs = self.pretraindatalayers(inputs)

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
        self.pretraindatalayers.eval()
        self.network.eval()

        # Run forward
        inputs = self.pretraindatalayers(inputs)

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
        self.pretraindatalayers = self.pretraindatalayers.cuda()
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.pretraindatalayers = self.pretraindatalayers.cpu()
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)

