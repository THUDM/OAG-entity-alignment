from os.path import join
import os
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset

import utils
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class ProcessedRNNInputDataset(Dataset):

    def __init__(self, entity_type, role):
        data_dir = join(settings.OUT_DIR, entity_type, "rnn")
        fname = "{}_rnn_{}.pkl".format(entity_type, role)
        data_dict = utils.load_large_obj(data_dir, fname)
        self.x1_seq1 = np.array(data_dict["x1_seq1"], dtype=int)
        self.x1_seq2 = np.array(data_dict["x1_seq2"], dtype=int)
        self.x2_seq1 = np.array(data_dict["x2_seq1"], dtype=int)
        self.x2_seq2 = np.array(data_dict["x2_seq2"], dtype=int)
        self.y = np.array(data_dict["y"], dtype=int)

        self.N = len(self.y)

        self.x1_seq1 = torch.from_numpy(self.x1_seq1)
        self.x1_seq2 = torch.from_numpy(self.x1_seq2)
        self.x2_seq1 = torch.from_numpy(self.x2_seq1)
        self.x2_seq2 = torch.from_numpy(self.x2_seq2)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1_seq1[idx], self.x2_seq1[idx], self.x1_seq2[idx], self.x2_seq2[idx], self.y[idx]


class ProcessedCNNInputDataset(Dataset):

    def __init__(self, entity_type, role):
        data_dir = join(settings.OUT_DIR, entity_type, "cnn")
        fname = "{}_cnn_{}.pkl".format(entity_type, role)
        data_dict = utils.load_large_obj(data_dir, fname)
        self.x1 = np.array(data_dict["x1"], dtype="float32")
        self.x2 = np.array(data_dict["x2"], dtype="float32")
        self.y = np.array(data_dict["y"], dtype=int)

        self.N = len(self.y)

        self.x1 = torch.from_numpy(self.x1)
        self.x2 = torch.from_numpy(self.x2)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


class PairedSubgraphDataset(Dataset):
    def __init__(self, seed, shuffle, role="train", entity_type="author"):
        self.file_dir = join(settings.OUT_DIR, entity_type, "hgat")

        # load subgraphs
        logger.info('loading adjs...')
        self.graphs = np.load(join(self.file_dir, 'adjacency_matrix_{}.npy'.format(role))).astype(np.float32)
        logger.info('adjs loaded')

        # add self-loop
        identity = np.identity(self.graphs.shape[1]).astype(np.bool_)
        self.graphs += identity
        self.graphs[self.graphs != 0] = True
        # self.graphs = self.graphs.astype(np.dtype('B'))
        logger.info('graph processed.')
        self.ego_size = self.graphs.shape[1]

        # load labels
        self.labels = np.load(os.path.join(self.file_dir, "label_{}.npy".format(role)))
        self.labels = self.labels.astype(np.long)
        logger.info("labels loaded!")

        # load vertices
        self.vertices = np.load(join(self.file_dir, 'vertex_id_{}.npy'.format(role)))
        logger.info('vertices loaded')

        # load vertex types
        self.vertex_types = np.load(os.path.join(self.file_dir, 'vertex_types_{}.npy'.format(role)))
        logger.info('vertex types loaded')

        self.x_stat = np.load(join(self.file_dir, "..", "{}_sim_stat_features_{}.npy".format(entity_type, role))).astype(np.float32)

        if shuffle:
            self.graphs, self.labels, self.vertices, self.vertex_types, self.x_stat = \
                sklearn.utils.shuffle(
                    self.graphs, self.labels, self.vertices, self.vertex_types, self.x_stat,
                    random_state=seed
                )

        input_dir = join(settings.DATA_DIR, "author")

        vertex_features = np.load(join(input_dir, "large_cross_graph_node_emb.npy"))
        vertex_features = np.concatenate((vertex_features, np.zeros((2, vertex_features.shape[1]))), axis=0)
        self.node_feature_dim = vertex_features.shape[1]

        node_list = []
        with open(join(input_dir, "large_cross_graph_nodes_list.txt")) as rf:
            for i, line in enumerate(rf):
                node_list.append(line.strip())
        self.id2idx = {n: i for i, n in enumerate(node_list)}
        node_to_idx_func = lambda x: self.id2idx.get(x, len(node_list))

        self.vertices = np.array(list(map(node_to_idx_func, self.vertices.flatten())),
                                 dtype=np.long).reshape(self.vertices.shape)  # convert to idx
        # print("vertices", self.vertices)
        self.node_features = torch.FloatTensor(vertex_features)
        self.vertex_types = torch.FloatTensor(self.vertex_types)

        self.N = len(self.graphs)
        logger.info("%d pair ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

    def get_embedding(self):
        return self.node_features

    def get_node_input_feature_dim(self):
        return self.node_feature_dim

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.vertices[idx], self.vertex_types[idx], self.x_stat[idx]
