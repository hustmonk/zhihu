from shell.datautil import *
from shell.argsconfig import *
from shell.Reader import Reader
from shell import utils
from shell import vector
import torch, json

iter_counter = 0

def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        global iter_counter
        iter_counter += 1

        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | iter = %d |' %
                        (global_stats['epoch'], idx, len(data_loader), iter_counter) +
                        'loss = %.4f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
            if torch.cuda.is_available() == False:
                break

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

def validate(args, data_loader, model, epoch):
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

    logger.info('dev: Epoch = %d | precision = %.4f | examples = %d | valid time = %.2f (s)' %
                (epoch, right, len(targets), eval_time.time()))
    return right

if __name__ == "__main__":
    args = initargs()

    settrain_defaults(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    logger = setlogger(args.log_file, args.checkpoint)

    train_exs = loadxml(os.path.join(args.data_dir, args.train_file))
    dev_exs = loadxml(os.path.join(args.data_dir, args.dev_file))

    logger.info("train:%d dev:%d" % (len(train_exs), len(dev_exs)))
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    word_dict = build_word_dict(train_exs + dev_exs)

    if args.pretrained:
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        model, start_epoch = Reader.load(args.pretrained)
    else:
        logger.info('Training model from scratch...')
        model = Reader(args)
    model.load_pretrained_dict(word_dict)

    if args.cuda:
        model.cuda()

    logger.info('=' * 60)
    psum = 0
    for (name, p) in model.network.named_parameters():
        psum += p.nelement()
    logger.info('Network total parameters ' + str(psum))
    logger.info('=' * 60)

    train_dataset = ReaderDataset(train_exs, model)
    dev_dataset = ReaderDataset(train_exs, model)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers=2,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        num_workers=2,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    notimprove_num = 0

    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch
        model.set_training(True)
        # Train
        train(args, train_loader, model, stats)

        model.set_training(False)

        result = validate(args, dev_loader, model, stats['epoch'])

        if result > stats['best_valid']:
            logger.info('Best valid: %.4f -> %.4f (epoch %d, %d updates)' %
                        (stats['best_valid'], result,
                         stats['epoch'], model.updates))
            model.save(args.model_file, epoch)
            stats['best_valid'] = result

