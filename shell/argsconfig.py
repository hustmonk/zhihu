import argparse
import argparse
import logging
import os, sys
import subprocess
logger = logging.getLogger(__name__)
from pathlib import PosixPath

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def initargs():
    WORK_SPACE = PosixPath(__file__).absolute().parents[1].as_posix()

    parser = argparse.ArgumentParser(
        'Document Reader', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--augmentation-data-weight', type=float, default=0.5,
                             help='weight of other argmentation datas')
    runtime.add_argument('--stop-early', type='bool', default=True,
                         help='if 7 epoch f1 didnt improve, the model stop')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=2,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=30,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--display-iter', type=int, default=10,
                         help='Batch size during validation/testing')
    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=WORK_SPACE + "/models/",
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=WORK_SPACE + "/data/",
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='train-data.xml', help='train file')
    files.add_argument('--dev-file', type=str,
                       default='dev-data.xml', help='dev file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=True,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--checkpoint-epoch', type='bool', default=False,
                           help='Save model + optimizer state with different file after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')

    pretrainedfiles = parser.add_argument_group('Pretrained File')

    pretrainedfiles.add_argument('--embedding-file', type=str,
                       default=WORK_SPACE + '/data/glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    pretrainedfiles.add_argument('--glove-embedding-dim', type=int, default= 300)
    pretrainedfiles.add_argument('--embedding-dim', type=int, default= 900)

    pretrainedfiles.add_argument('--cove-file', type=str,
                       default=WORK_SPACE + '/data/wmtlstm-b142a7f2.pth',
                       help='Space-separated pretrained cove file')

    model = parser.add_argument_group('Reader Model Architecture')
    model.add_argument('--hidden-size', type=int, default=125,
                       help='Hidden size of RNN units')
    model.add_argument('--learning-rate', type=float, default=2e-3,
                       help='Learning rate for SGD only')
    model.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')
    model.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    model.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    args = parser.parse_args()
    return args

def settrain_defaults(args):
    subprocess.call(['mkdir', '-p', args.model_dir])
    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-%H%M%S-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.model_file_ema = os.path.join(args.model_dir, args.model_name + '.ema.mdl')
    args.model_file_checkpoint = os.path.join(args.model_dir, args.model_name + '.checkpoint.mdl')

    return args

def setlogger(log_file=None, checkpoint=False):
    subprocess.call(['mkdir', '-p', 'log'])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        if checkpoint:
            logfile = logging.FileHandler(log_file, 'a')
        else:
            logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

    try:
        GIT_BRANCH = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'])
        GIT_REVISION = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        GIT_BRANCH = "Unknown"
        GIT_REVISION = "Unknown"

    logger.info('COMMAND: [ nohup python %s & ], GIT_REVISION: [%s] [%s]'
                % (' '.join(sys.argv), GIT_BRANCH.strip(), GIT_REVISION.strip()))
    return logger