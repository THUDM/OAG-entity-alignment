from os.path import join
import os
import sys
import time
import argparse
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from data_loader import ProcessedRNNInputDataset
from models import BiLSTM
from utils import ChunkSampler
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)  # include timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--delta-seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--embedding-size', type=int, default=128, help="Embeding size for LSTM layer")
parser.add_argument('--hidden-size', type=int, default=32, help="Hidden size for LSTM layer")
parser.add_argument('--max-sequence-length', type=int, default=10, help="Max sequence length for raw sequences")
parser.add_argument('--max-key-sequence-length', type=int, default=5, help="Max key sequence length for key sequences")
parser.add_argument('--batch', type=int, default=32, help="Batch size")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--entity-type', type=str, default="venue", help="entity type to match")

parser.add_argument('--check-point', type=int, default=5, help="Check point")
parser.add_argument('--multiple', type=int, default=16, help="decide how many times to multiply a scalar input")
parser.add_argument('--n-try', type=int, default=5, help="Repeat Times")
parser.add_argument('--n-seq', type=int, default=2, help="Repeat Times")

args = parser.parse_args()


def evaluate(epoch, loader, model, thr=None, return_best_thr=False, args=args):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []

    for ibatch, batch in enumerate(loader):
        labels = batch[-1]
        bs = len(labels)

        if args.cuda:
            batch = [data.cuda() for data in batch]
            labels = labels.cuda()
        output = model(batch[0], batch[1], batch[2], batch[3])
        loss_batch = F.nll_loss(output, labels)
        loss += bs * loss_batch.item()
        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += len(labels)

    model.train()

    if thr is not None:
        print("using threshold %.4f" % thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                loss / total, auc, prec, rec, f1)

    if return_best_thr:  # valid
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))

        return best_thr, [loss / total, auc, prec, rec, f1]
    else:
        return None, [loss / total, auc, prec, rec, f1]


def train(epoch, train_loader, valid_loader, test_loader, model, optimizer, args=args):
    model.train()

    loss = 0.
    total = 0.

    for i_batch, batch in enumerate(train_loader):
        bs = batch[-1].shape[0]
        labels = batch[-1]

        if args.cuda:
            batch = [data.cuda() for data in batch]
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(batch[0], batch[1], batch[2], batch[3])
        loss_train = F.nll_loss(output, labels)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss epoch %d: %f", epoch, loss / total)

    metrics_val = None
    metrics_test = None
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        best_thr, metrics_val = evaluate(epoch, valid_loader, model, return_best_thr=True, args=args)
        logger.info('eval on test data!...')
        _, metrics_test = evaluate(epoch, test_loader, model, thr=best_thr, args=args)

    return metrics_val, metrics_test


def train_one_time(args, wf, repeat_seed):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    dataset = ProcessedRNNInputDataset(args.entity_type, "train")
    dataset_valid = ProcessedRNNInputDataset(args.entity_type, "valid")
    dataset_test = ProcessedRNNInputDataset(args.entity_type, "test")

    N = len(dataset)
    N_valid = len(dataset_valid)
    N_test = len(dataset_test)
    print("n_train", N)
    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(N, 0))
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch,
                              sampler=ChunkSampler(N_valid, 0))
    test_loader = DataLoader(dataset_test, batch_size=args.batch,
                             sampler=ChunkSampler(N_test, 0))

    np.random.seed(args.seed + repeat_seed)
    torch.manual_seed(args.seed + repeat_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + repeat_seed)

    if args.entity_type != "author":
        pretrain_emb = torch.randn(size=(settings.MAX_WORD_TOKEN_NUM + 1, 128))
    else:
        raise NotImplementedError

    model = BiLSTM(vocab_size=settings.MAX_WORD_TOKEN_NUM,
                   pretrain_emb=pretrain_emb,
                   embedding_size=args.embedding_size,
                   hidden_size=args.hidden_size,
                   dropout=args.dropout,
                   use_seq_num=args.n_seq)
    print(model)
    n_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of paras:", n_paras)

    if args.cuda:
        model.cuda()
    model = model.float()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_total = time.time()
    logger.info("training...")

    loss_val_min = None
    best_test_metric = None
    best_epoch = -1
    model_dir = join(settings.OUT_DIR, args.entity_type, "rnn-models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        metrics = train(epoch, train_loader, valid_loader, test_loader, model, optimizer, args=args)

        metrics_val, metrics_test = metrics
        if metrics_val is not None:
            if loss_val_min is None or metrics_val[0] < loss_val_min:
                loss_val_min = metrics_val[0]
                best_test_metric = metrics_test
                best_model = model
                best_epoch = epoch
                torch.save(best_model.state_dict(),
                           join(model_dir, "rnn-match-best-now-try-{}.mdl".format(repeat_seed)))

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    print("best epoch", best_epoch)
    print("min_val_loss", loss_val_min, "best test metrics", best_test_metric[1:])
    wf.write(
        "min valid loss {:.4f}, best test metrics: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n\n".format(
            loss_val_min, best_test_metric[1] * 100, best_test_metric[2] * 100, best_test_metric[3] * 100,
            best_test_metric[4] * 100
        ))


def main(args):
    model_dir = join(settings.OUT_DIR, args.entity_type)
    os.makedirs(model_dir, exist_ok=True)
    wf = open(join(model_dir, "{}_rnn_results_n_seq_{}_seqs_len_{}_{}.txt".
                   format(args.entity_type, args.n_seq, args.max_sequence_length, args.max_key_sequence_length)), "w")
    for t in range(args.n_try):
        train_one_time(args, wf, t)
        wf.flush()
    wf.write(json.dumps(vars(args)) + "\n")
    wf.close()


def calc_avg_metrics(args):
    model_dir = join(settings.OUT_DIR, args.entity_type)
    metrics = []
    with open(join(model_dir, "{}_rnn_results_n_seq_{}_seqs_len_{}_{}.txt".
            format(args.entity_type, args.n_seq, args.max_sequence_length, args.max_key_sequence_length))) as rf:
        for i, line in enumerate(rf):
            line = line.strip()
            if i % 2 == 0 and i < 10:
                items = line.split(":")[2:]
                items = [float(x.split(",")[0].strip()) for x in items]
                metrics.append(items)
    print(np.mean(np.array(metrics), axis=0))


if __name__ == "__main__":
    print("args", args)
    main(args=args)
    # calc_avg_metrics(args)
    logger.info("done")
