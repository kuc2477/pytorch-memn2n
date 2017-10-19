#!/usr/bin/env python3
import argparse
import torch
from model import MemN2N
from train import train
from data import BabiQA
import utils


parser = argparse.ArgumentParser(
    'End-to-End Memory Network PyTorch Implementation'
)
parser.add_argument('--vocabulary-size', type=int, default=200)
parser.add_argument('--embedding-size', type=int, default=50)
parser.add_argument('--sentence-size', type=int, default=20)
parser.add_argument('--memory-size', type=int, default=30)
parser.add_argument('--hops', type=int, default=3)
parser.add_argument(
    '--weight-tying-scheme',
    choices=MemN2N.WEIGHT_TYING_SCHEMES,
    default=MemN2N.ADJACENT
)
parser.add_argument('--babi-dataset-name', default='en-10k')
parser.add_argument(
    '--babi-tasks', type=int, default=[i+1 for i in range(20)], nargs='+'
)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--test-size', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--weight-decay', type=float, default=1e-04)
parser.add_argument('--grad-clip-norm', type=float, default=5.)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--lr-decay', type=float, default=.1)
parser.add_argument('--lr-decay-epochs', type=int, default=[30, 50, 80],
                    nargs='+')

parser.add_argument('--checkpoint-interval', type=int, default=5000)
parser.add_argument('--eval-log-interval', type=int, default=100)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--gradient-log-interval', type=int, default=50)
parser.add_argument('--model-dir', default='./checkpoints')
parser.add_argument('--dataset-dir', default='./datasets')

resume_command = parser.add_mutually_exclusive_group()
resume_command.add_argument('--resume-best', action='store_true')
resume_command.add_argument('--resume-latest', action='store_true')
parser.add_argument('--best', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_false', dest='test')
main_command.add_argument('--test', action='store_true', dest='test')


if __name__ == "__main__":
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    dataset_config = dict(
        dataset_name=args.babi_dataset_name,
        tasks=args.babi_tasks,
        vocabulary_size=args.vocabulary_size,
        sentence_size=args.sentence_size,
        sentence_number=args.memory_size,
        path=args.dataset_dir,
    )
    train_dataset = BabiQA(**dataset_config, train=True)
    test_dataset = BabiQA(**dataset_config, train=False,
                          vocabulary=train_dataset.vocabulary)

    memn2n = MemN2N(
        vocabulary_hash=train_dataset.vocabulary_hash,
        vocabulary_size=train_dataset.vocabulary_size,
        embedding_size=args.embedding_size,
        sentence_size=args.sentence_size,
        memory_size=args.memory_size,
        hops=args.hops,
        weight_tying_scheme=args.weight_tying_scheme,
    )

    # initialize the weights.
    utils.gaussian_intiailize(memn2n)

    # prepare cuda if needed.
    if cuda:
        memn2n.cuda()

    # run the given command.
    if args.test:
        utils.load_checkpoint(memn2n, args.model_dir, best=True)
        utils.validate(
            memn2n, test_dataset, test_size=args.test_size,
            cuda=cuda, verbose=True
        )
    else:
        train(
            memn2n,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_dir=args.model_dir,
            collate_fn=BabiQA.collate_fn,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_epochs=args.lr_decay_epochs,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            batch_size=args.batch_size,
            test_size=args.test_size,
            epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            eval_log_interval=args.eval_log_interval,
            gradient_log_interval=args.gradient_log_interval,
            loss_log_interval=args.loss_log_interval,
            resume_best=args.resume_best,
            resume_latest=args.resume_latest,
            cuda=cuda,
        )
