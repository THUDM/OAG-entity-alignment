from os.path import join
import argparse
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn import svm

import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--entity-type', type=str, default="author", help="entity type to match")
args = parser.parse_args()


def fit_and_test_svm(entity_type=args.entity_type):
    cur_out_dir = join(settings.OUT_DIR, entity_type)

    x_train = np.load(join(cur_out_dir, "{}_sim_stat_features_train.npy".format(entity_type)))
    x_test = np.load(join(cur_out_dir, "{}_sim_stat_features_test.npy".format(entity_type)))
    y_train = np.load(join(cur_out_dir, "{}_labels_train.npy".format(entity_type)))
    y_test = np.load(join(cur_out_dir, "{}_labels_test.npy".format(entity_type)))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=42)
    clf = svm.SVC(verbose=True, probability=True, kernel="rbf")
    clf.fit(x_train, y_train)

    x_test = scaler.transform(x_test)
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_score[:, 1])
    logger.info("\nevaluating %s alignment...", entity_type)
    logger.info('pred results %.4f %.4f %.4f', prec, rec, f1)
    logger.info('auc score %.4f', auc)


if __name__ == "__main__":
    fit_and_test_svm(entity_type=args.entity_type)
    logger.info("done")
