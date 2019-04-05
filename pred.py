from shell.datautil import *
from shell.argsconfig import *
from shell.Reader import Reader
from shell import utils
from shell import vector
import torch, json, sys

iter_counter = 0


dev_exs = loadxml("data/dev-data.xml")
word_dict = build_word_dict(dev_exs)

pretrained = sys.argv[1]

model, start_epoch = Reader.load(pretrained)
model.load_pretrained_dict(word_dict)

args = model.args

dev_dataset = ReaderDataset(dev_exs, model)
data_loader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    collate_fn=vector.batchify,
    pin_memory=args.cuda,
)

eval_time = utils.Timer()
# Run through examples
ids = []
preds = []
targets = []
for ex in data_loader:
    ids, inputs, target = ex
    pred = model.predict(inputs)
    preds += pred
    targets += target.numpy().tolist()
    if torch.cuda.is_available() == False:
        break

right = 1.0 * sum([1 for (p, t) in zip(preds, targets) if p == t]) / len(preds)

print('dev: precision = %.4f | examples = %d | valid time = %.2f (s)' %
            (right, len(targets), eval_time.time()))
