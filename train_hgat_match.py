import argparse
from os.path import join
import os
import time
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from data_loader import PairedSubgraphDataset
from models import MatchBatchHGAT
from utils import ChunkSampler
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="32,8",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8,1", help="Heads in each layer, splitted with comma")
parser.add_argument('--batch', type=int, default=64, help="Batch size")
parser.add_argument('--check-point', type=int, default=5, help="Check point")
parser.add_argument('--n-type-nodes', type=int, default=3, help="the number of different types of nodes")
parser.add_argument('--instance-normalization', action='store_true', default=True,
                    help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--train-ratio', type=float, default=8/9, help="Training ratio (0, 100)")
parser.add_argument('--n-try', type=int, default=1, help="Repeat Times")
parser.add_argument('--entity-type', type=str, default="author", help="entity type to match")

args = parser.parse_args()


def evaluate(epoch, loader, model, node_init_features, thr=None, return_best_thr=False, args=args):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []

    for i_batch, batch in enumerate(loader):
        graph, labels, vertices, v_types_orig, x_stat = batch

        features = torch.FloatTensor(node_init_features[vertices])

        bs = len(labels)
        n = vertices.shape[1]

        v_types = np.zeros((bs, n, args.n_type_nodes))
        v_types_orig = v_types_orig.numpy()
        for ii in range(bs):
            for vv in range(n):
                idx = int(v_types_orig[ii, vv])
                v_types[ii, vv, idx] = 1
        v_types = torch.Tensor(v_types)  # bs x n x n_node_type

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            features = features.cuda()
            v_types = v_types.cuda()
            x_stat = x_stat.cuda()

        output = model(features, graph, v_types, x_stat)
        loss_batch = F.nll_loss(output, labels)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
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


def train(epoch, train_loader, valid_loader, test_loader, model, optimizer, node_init_features, args=args):
    model.train()

    loss = 0.
    total = 0.

    for i_batch, batch in enumerate(train_loader):
        graph, labels, vertices, v_types_orig, x_stat = batch
        features = torch.FloatTensor(node_init_features[vertices])
        bs = len(labels)
        n = vertices.shape[1]

        v_types = np.zeros((bs, n, args.n_type_nodes))
        v_types_orig = v_types_orig.numpy()
        for ii in range(bs):
            for vv in range(n):
                idx = int(v_types_orig[ii, vv])
                v_types[ii, vv, idx] = 1
        v_types = torch.Tensor(v_types)  # bs x n x n_node_type

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            features = features.cuda()
            v_types = v_types.cuda()
            x_stat = x_stat.cuda()

        optimizer.zero_grad()
        output = model(features, graph, v_types, x_stat)

        loss_train = F.nll_loss(output, labels)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()

    metrics_val = None
    metrics_test = None
    logger.info("train loss epoch %d: %f", epoch, loss / total)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        best_thr, metrics_val = evaluate(epoch, valid_loader, model, node_init_features, return_best_thr=True, args=args)
        logger.info('eval on test data!...')
        _, metrics_test = evaluate(epoch, test_loader, model, node_init_features, thr=best_thr, args=args)

    return metrics_val, metrics_test


def train_one_time(args, wf, repeat_seed):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed + repeat_seed)
    torch.manual_seed(args.seed + repeat_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + repeat_seed)

    dataset = PairedSubgraphDataset(seed=args.seed, shuffle=True, role="train")
    dataset_test = PairedSubgraphDataset(seed=args.seed, shuffle=False, role="test")
    N = len(dataset)
    n_train = int(N*args.train_ratio)
    n_valid = N - n_train
    print("n_train", n_train)

    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(n_train, 0))
    valid_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(n_valid, n_train))
    test_loader = DataLoader(dataset_test, batch_size=args.batch,
                             sampler=ChunkSampler(len(dataset_test), 0))

    input_feature_dim = dataset.get_node_input_feature_dim()
    n_units = [input_feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = MatchBatchHGAT(n_type_nodes=args.n_type_nodes,
                           n_units=n_units,
                           n_head=n_heads[0],
                           dropout=args.dropout,
                           attn_dropout=args.attn_dropout,
                           instance_normalization=args.instance_normalization)
    node_init_features = dataset.get_embedding().numpy()

    if args.cuda:
        model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    logger.info("training...")

    auc_val_max = None
    best_test_metric = None
    best_epoch = -1
    model_dir = join(settings.OUT_DIR, args.entity_type, "hgat-models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        metrics = train(epoch, train_loader, valid_loader, test_loader, model, optimizer, node_init_features, args=args)
        metrics_val, metrics_test = metrics
        if metrics_val is not None:
            # if loss_val_min is None or metrics_val[0] < loss_val_min:
            if auc_val_max is None or metrics_val[1] > auc_val_max:
                # loss_val_min = metrics_val[0]
                auc_val_max = metrics_val[1]
                best_test_metric = metrics_test
                best_model = model
                best_epoch = epoch
                torch.save(best_model.state_dict(),
                           join(model_dir, "hgat-match-best-now-try-{}.mdl".format(repeat_seed)))

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    print("best epoch", best_epoch)
    print("max_auc_val", auc_val_max, "best test metrics", best_test_metric[1:])

    wf.write(
        "max valid auc {:.4f}, best test metrics: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n\n".format(
            auc_val_max, best_test_metric[1] * 100, best_test_metric[2] * 100, best_test_metric[3] * 100,
            best_test_metric[4] * 100
        ))


def main(args):
    model_dir = join(settings.OUT_DIR, args.entity_type)
    os.makedirs(model_dir, exist_ok=True)

    wf = open(join(model_dir, "{}_hgat_results.txt".format(args.entity_type)), "w")
    for t in range(args.n_try):
        train_one_time(args, wf, t)
        wf.flush()
    wf.write(json.dumps(vars(args)) + "\n")
    wf.close()


def calc_avg_metrics(args):
    model_dir = join(settings.OUT_DIR, args.entity_type)
    metrics = []
    with open(join(model_dir, "{}_hgat_results.txt".format(args.entity_type))) as rf:
        for i, line in enumerate(rf):
            line = line.strip()
            if i % 2 == 0 and i < 10:
                items = line.split(":")[2:]
                items = [float(x.split(",")[0].strip()) for x in items]
                metrics.append(items)
    print(np.mean(np.array(metrics), axis=0))


if __name__ == '__main__':
    print("args", args)
    main(args=args)
    calc_avg_metrics(args)
