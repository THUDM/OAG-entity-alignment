import os
from os.path import join
import warnings
from tqdm import tqdm
from collections import Counter
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import utils
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

warnings.simplefilter(action='ignore', category=FutureWarning)


def gen_short_text_pair_sim_features(e1, e2, tfidf_vec):
    e1 = utils.normalize_text(e1)
    e2 = utils.normalize_text(e2)
    tokens1 = e1.split()
    tokens2 = e2.split()

    overlap = Counter(tokens1) & Counter(tokens2)
    jaccard = sum(overlap.values()) / (len(tokens1) + len(tokens2) - sum(overlap.values()))

    tfidf_vectors = tfidf_vec.transform([e1, e2])
    cos_sim = cosine_similarity(tfidf_vectors, tfidf_vectors)
    tfidf_cos_sim = cos_sim[0, 1]

    return [jaccard, tfidf_cos_sim]


def gen_short_texts_sim_stat_features(entity_type="aff"):
    cur_data_dir = join(settings.DATA_DIR, entity_type)

    pairs_train = utils.load_json(cur_data_dir, "{}_alignment_train_pairs.json".format(entity_type))
    pairs_test = utils.load_json(cur_data_dir, "{}_alignment_test_pairs.json".format(entity_type))

    text_corpus = set()
    for pair in pairs_train + pairs_test:
        text_corpus.add(utils.normalize_text(pair["{}1".format(entity_type)]))
        text_corpus.add(utils.normalize_text(pair["{}2".format(entity_type)]))

    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(list(text_corpus))
    print(tfidf_vec.vocabulary_)
    print(tfidf_vec.idf_)

    feat_train = []
    labels_train = []
    for pair in tqdm(pairs_train):
        cur_vec = gen_short_text_pair_sim_features(pair["{}1".format(entity_type)], pair["{}2".format(entity_type)], tfidf_vec)
        feat_train.append(cur_vec)
        labels_train.append(pair["label"])

    feat_test = []
    labels_test = []
    for pair in tqdm(pairs_test):
        cur_vec = gen_short_text_pair_sim_features(pair["{}1".format(entity_type)], pair["{}2".format(entity_type)], tfidf_vec)
        feat_test.append(cur_vec)
        labels_test.append(pair["label"])

    cur_out_dir = join(settings.OUT_DIR, entity_type)
    os.makedirs(cur_out_dir, exist_ok=True)
    np.save(join(cur_out_dir, "{}_sim_stat_features_train.npy".format(entity_type)), np.array(feat_train))
    np.save(join(cur_out_dir, "{}_sim_stat_features_test.npy".format(entity_type)), np.array(feat_test))
    np.save(join(cur_out_dir, "{}_labels_train.npy".format(entity_type)), labels_train)
    np.save(join(cur_out_dir, "{}_labels_test.npy".format(entity_type)), labels_test)


def gen_each_author_pair_struct_feature(pids1, pids2, pids_map):
    overlap = Counter(pids1) & Counter(pids2)
    jaccard = sum(overlap.values()) / (len(pids1) + len(pids2) - sum(overlap.values()))

    vids1 = ["/".join(pids_map[x].split("/")[:-1]) for x in pids1]
    vids2 = ["/".join(pids_map[x].split("/")[:-1]) for x in pids2]
    print("vids1", vids1)
    print("vids2", vids2)

    overlap2 = Counter(vids1) & Counter(vids2)
    jaccard2 = sum(overlap2.values()) / (len(vids1) + len(vids2) - sum(overlap2.values()))
    return [jaccard, jaccard2]


def gen_author_sim_struct_features():
    cur_data_dir = join(settings.DATA_DIR, "author")

    pairs_train = utils.load_json(cur_data_dir, "author_alignment_train_pairs.json")
    pairs_test = utils.load_json(cur_data_dir, "author_alignment_test_pairs.json")

    paper_aid_to_did = utils.load_json(cur_data_dir, "aminer_to_dblp_paper_map.json")
    aminer_author_info = utils.load_json(cur_data_dir, "aminer_ego_author_attr_dict.json")
    dblp_author_info = utils.load_json(cur_data_dir, "dblp_ego_author_attr_dict.json")

    feat_train = []
    feat_test = []
    for pair in tqdm(pairs_train):
        aid = pair["aminer"]
        name_d = pair["dblp"]
        cur_vec = gen_each_author_pair_struct_feature(aminer_author_info[aid]["pubs"], dblp_author_info[name_d]["pubs"], paper_aid_to_did)
        feat_train.append(cur_vec)

    for pair in tqdm(pairs_test):
        aid = pair["aminer"]
        name_d = pair["dblp"]
        cur_vec = gen_each_author_pair_struct_feature(aminer_author_info[aid]["pubs"], dblp_author_info[name_d]["pubs"], paper_aid_to_did)
        feat_test.append(cur_vec)

    out_dir = join(settings.OUT_DIR, "author")
    os.makedirs(out_dir, exist_ok=True)
    entity = "author"
    np.save(join(out_dir, "{}_sim_stat_features_train.npy".format(entity)), np.array(feat_train))
    np.save(join(out_dir, "{}_sim_stat_features_test.npy".format(entity)), np.array(feat_test))
    np.save(join(out_dir, "{}_labels_train.npy".format(entity)), [x["label"] for x in pairs_train])
    np.save(join(out_dir, "{}_labels_test.npy".format(entity)), [x["label"] for x in pairs_test])


def build_tokenizer(entity_type, pairs_train, pairs_test):
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(texts):
        for text in texts:
            yield tokenizer(text)

    text_corpus = set()
    for pair in tqdm(pairs_train + pairs_test):
        text_corpus.add(utils.normalize_text(pair["{}1".format(entity_type)]))
        text_corpus.add(utils.normalize_text(pair["{}2".format(entity_type)]))

    vocab = build_vocab_from_iterator(yield_tokens(list(text_corpus)))

    def text_pipeline(x):
        return vocab(tokenizer(x))

    return vocab, text_pipeline


def calc_keywords_seqs(x1, x2):
    N = len(x1)
    x1_keywords = []
    x2_keywords = []
    for i in tqdm(range(N)):
        item1 = x1[i].tolist()
        item2 = x2[i].tolist()
        overlap = Counter(item1) & Counter(item2)
        item1_new = []
        item2_new = []
        for w in item1:
            if w in overlap:
                item1_new.append(w)
        for w in item2:
            if w in overlap:
                item2_new.append(w)
        x1_keywords.append(torch.LongTensor(item1_new))
        x2_keywords.append(torch.LongTensor(item2_new))
    return x1_keywords, x2_keywords


def process_rnn_match_pair(entity_type, max_seq1_len=10, max_seq2_len=5, shuffle=True, seed=42):
    from keras.preprocessing.sequence import pad_sequences

    file_dir = join(settings.DATA_DIR, entity_type)

    pairs_train = utils.load_json(file_dir, "{}_alignment_{}_pairs.json".format(entity_type, "train"))
    pairs_test = utils.load_json(file_dir, "{}_alignment_{}_pairs.json".format(entity_type, "test"))

    vocab, text_pipeline = build_tokenizer(entity_type, pairs_train, pairs_test)

    x1 = []
    x2 = []
    labels = []
    for pair in tqdm(pairs_train):
        item1 = utils.normalize_text(pair["{}1".format(entity_type)])
        item2 = utils.normalize_text(pair["{}2".format(entity_type)])
        item1 = text_pipeline(item1)
        item2 = text_pipeline(item2)
        x1.append(torch.LongTensor(item1))
        x2.append(torch.LongTensor(item2))
        labels.append(pair["label"])

    # test
    x1_test = []
    x2_test = []
    labels_test = []
    for pair in tqdm(pairs_test):
        item1 = utils.normalize_text(pair["{}1".format(entity_type)])
        item2 = utils.normalize_text(pair["{}2".format(entity_type)])
        item1 = text_pipeline(item1)
        item2 = text_pipeline(item2)
        x1_test.append(torch.LongTensor(item1))
        x2_test.append(torch.LongTensor(item2))
        labels_test.append(pair["label"])

    x1_keywords, x2_keywords = calc_keywords_seqs(x1, x2)
    x1_test_keywords, x2_test_keywords = calc_keywords_seqs(x1_test, x2_test)
    vocab_size = len(vocab) + 2

    x1 = pad_sequences(x1, maxlen=max_seq1_len, value=len(vocab) + 1)
    x2 = pad_sequences(x2, maxlen=max_seq1_len, value=len(vocab) + 1)

    x1_keywords = pad_sequences(x1_keywords, maxlen=max_seq2_len, value=len(vocab) + 1)
    x2_keywords = pad_sequences(x2_keywords, maxlen=max_seq2_len, value=len(vocab) + 1)

    x1_test = pad_sequences(x1_test, maxlen=max_seq1_len, value=len(vocab) + 1)
    x2_test = pad_sequences(x2_test, maxlen=max_seq1_len, value=len(vocab) + 1)

    x1_test_keywords = pad_sequences(x1_test_keywords, maxlen=max_seq2_len, value=len(vocab) + 1)
    x2_test_keywords = pad_sequences(x2_test_keywords, maxlen=max_seq2_len, value=len(vocab) + 1)

    if shuffle:
        x1, x2, x1_keywords, x2_keywords, labels = sklearn.utils.shuffle(
            x1, x2, x1_keywords, x2_keywords, labels,
            random_state=seed
        )

    out_dir = join(settings.OUT_DIR, entity_type, "rnn")
    os.makedirs(out_dir, exist_ok=True)

    N = len(labels)

    if entity_type == "aff":
        n_train = int(N*0.75)
        n_valid = int(N*0.25)
    elif entity_type == "venue":
        n_train = int(N / 5 * 4)
        n_valid = int(N / 5)
    else:
        raise NotImplementedError

    train_data = {}
    train_data["x1_seq1"] = x1[:n_train]
    train_data["x1_seq2"] = x1_keywords[:n_train]
    train_data["x2_seq1"] = x2[:n_train]
    train_data["x2_seq2"] = x2_keywords[:n_train]
    train_data["y"] = labels[:n_train]
    train_data["vocab_size"] = vocab_size
    print("train labels", len(train_data["y"]))

    valid_data = {}
    valid_data["x1_seq1"] = x1[n_train:(n_train + n_valid)]
    valid_data["x1_seq2"] = x1_keywords[n_train:(n_train + n_valid)]
    valid_data["x2_seq1"] = x2[n_train:(n_train + n_valid)]
    valid_data["x2_seq2"] = x2_keywords[n_train:(n_train + n_valid)]
    valid_data["y"] = labels[n_train:(n_train + n_valid)]
    print("valid labels", len(valid_data["y"]))

    utils.dump_large_obj(train_data, out_dir, "{}_rnn_train.pkl".format(entity_type))
    utils.dump_large_obj(valid_data, out_dir, "{}_rnn_valid.pkl".format(entity_type))

    test_data = {}
    test_data["x1_seq1"] = x1_test
    test_data["x1_seq2"] = x1_test_keywords
    test_data["x2_seq1"] = x2_test
    test_data["x2_seq2"] = x2_test_keywords
    test_data["y"] = labels_test
    print("test labels", len(test_data["y"]))

    utils.dump_large_obj(test_data, out_dir, "{}_rnn_test.pkl".format(entity_type))


def process_rnn_author_match_pair(max_seq1_len=10, max_seq2_len=5, shuffle=True, seed=42):
    from keras.preprocessing.sequence import pad_sequences
    cur_data_dir = join(settings.DATA_DIR, "author")

    pairs_train = utils.load_json(cur_data_dir, "author_alignment_train_pairs.json")
    pairs_test = utils.load_json(cur_data_dir, "author_alignment_test_pairs.json")

    node_list = []
    with open(join(cur_data_dir, "large_cross_graph_nodes_list.txt")) as rf:
        for i, line in enumerate(rf):
            node_list.append(line.strip())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    aminer_ego_author_attr = utils.load_json(cur_data_dir, "aminer_ego_author_attr_dict.json")
    dblp_ego_author_attr = utils.load_json(cur_data_dir, "dblp_ego_author_attr_dict.json")

    aminer_aid_to_vids = utils.load_json(cur_data_dir, "aminer_aid_to_vids.json")
    dblp_aid_to_vids = utils.load_json(cur_data_dir, "dblp_aid_to_vids.json")

    x1 = []
    x2 = []
    x1_keywords = []
    x2_keywords = []
    for pair in tqdm(pairs_train):
        aid = pair["aminer"]
        did = pair["dblp"]
        pubs_a = aminer_ego_author_attr.get(aid, [])["pubs"][: max_seq1_len]
        pubs_d = dblp_ego_author_attr.get(did, [])["pubs"][: max_seq1_len]
        pubs_a_idx = [node_to_idx[x + "-pp"] for x in pubs_a]
        pubs_d_idx = [node_to_idx[x + "-pp"] for x in pubs_d]
        x1.append(pubs_a_idx)
        x2.append(pubs_d_idx)

        vids_a = aminer_aid_to_vids.get(aid, [])[: max_seq2_len]
        vids_d = dblp_aid_to_vids.get(aid, [])[: max_seq2_len]
        vids_a_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_a]
        vids_d_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_d]
        x1_keywords.append(vids_a_idx)
        x2_keywords.append(vids_d_idx)

    # test
    x1_test = []
    x2_test = []
    x1_keywords_test = []
    x2_keywords_test = []
    for pair in tqdm(pairs_test):
        aid = pair["aminer"]
        did = pair["dblp"]
        pubs_a = aminer_ego_author_attr.get(aid, [])["pubs"][: max_seq1_len]
        pubs_d = dblp_ego_author_attr.get(did, [])["pubs"][: max_seq1_len]
        pubs_a_idx = [node_to_idx[x + "-pp"] for x in pubs_a]
        pubs_d_idx = [node_to_idx[x + "-pp"] for x in pubs_d]
        x1_test.append(pubs_a_idx)
        x2_test.append(pubs_d_idx)

        vids_a = aminer_aid_to_vids.get(aid, [])[: max_seq2_len]
        vids_d = dblp_aid_to_vids.get(aid, [])[: max_seq2_len]
        vids_a_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_a]
        vids_d_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_d]
        x1_keywords_test.append(vids_a_idx)
        x2_keywords_test.append(vids_d_idx)

    n_nodes = len(node_list)

    x1 = pad_sequences(x1, maxlen=max_seq1_len, value=n_nodes + 1)
    x2 = pad_sequences(x2, maxlen=max_seq1_len, value=n_nodes + 1)

    x1_keywords = pad_sequences(x1_keywords, maxlen=max_seq2_len, value=n_nodes + 1)
    x2_keywords = pad_sequences(x2_keywords, maxlen=max_seq2_len, value=n_nodes + 1)

    x1_test = pad_sequences(x1_test, maxlen=max_seq1_len, value=n_nodes + 1)
    x2_test = pad_sequences(x2_test, maxlen=max_seq1_len, value=n_nodes + 1)

    x1_test_keywords = pad_sequences(x1_keywords_test, maxlen=max_seq2_len, value=n_nodes + 1)
    x2_test_keywords = pad_sequences(x2_keywords_test, maxlen=max_seq2_len, value=n_nodes + 1)

    labels = [x["label"] for x in pairs_train]
    labels_test = [x["label"] for x in pairs_test]

    if shuffle:
        x1, x2, x1_keywords, x2_keywords, labels = sklearn.utils.shuffle(
            x1, x2, x1_keywords, x2_keywords, labels,
            random_state=seed
        )

    out_dir = join(settings.OUT_DIR, "author", "rnn")
    os.makedirs(out_dir, exist_ok=True)

    N = len(pairs_train)
    n_train = int(N / 9 * 8)
    n_valid = int(N / 9)

    train_data = {}
    train_data["x1_seq1"] = x1[:n_train]
    train_data["x1_seq2"] = x1_keywords[:n_train]
    train_data["x2_seq1"] = x2[:n_train]
    train_data["x2_seq2"] = x2_keywords[:n_train]
    train_data["y"] = labels[:n_train]
    train_data["vocab_size"] = n_nodes
    print("train labels", len(train_data["y"]))

    valid_data = {}
    valid_data["x1_seq1"] = x1[n_train:(n_train + n_valid)]
    valid_data["x1_seq2"] = x1_keywords[n_train:(n_train + n_valid)]
    valid_data["x2_seq1"] = x2[n_train:(n_train + n_valid)]
    valid_data["x2_seq2"] = x2_keywords[n_train:(n_train + n_valid)]
    valid_data["y"] = labels[n_train:(n_train + n_valid)]
    print("valid labels", len(valid_data["y"]))

    utils.dump_large_obj(train_data, out_dir, "author_rnn_train.pkl")
    utils.dump_large_obj(valid_data, out_dir, "author_rnn_valid.pkl")

    test_data = {}

    test_data["x1_seq1"] = x1_test
    test_data["x1_seq2"] = x1_test_keywords
    test_data["x2_seq1"] = x2_test
    test_data["x2_seq2"] = x2_test_keywords
    test_data["y"] = labels_test
    print("test labels", len(test_data["y"]))

    utils.dump_large_obj(test_data, out_dir, "author_rnn_test.pkl")


def complete_sentence_pair_to_matrix(t1, t2, mat_size, pretrain_emb=None):
    matrix = -np.ones((mat_size, mat_size))
    for i, word1 in enumerate(t1[: mat_size]):
        for j, word2 in enumerate(t2[: mat_size]):
            v = -1
            if word1 == word2:
                v = 1
            elif pretrain_emb is not None:
                v = cosine_similarity(pretrain_emb[word1].reshape(1, -1),
                                      pretrain_emb[word2].reshape(1, -1))[0][0]

            matrix[i][j] = v
    return matrix


def sentences_overlap_to_matrix(t1, t2, mat_size):
    overlap = set(t1).intersection(t2)
    new_seq_t1 = []
    new_seq_t2 = []
    for w in t1:
        if w in overlap:
            new_seq_t1.append(w)
    for w in t2:
        if w in overlap:
            new_seq_t2.append(w)

    matrix = -np.ones((mat_size, mat_size))
    for i, word1 in enumerate(t1[: mat_size]):
        for j, word2 in enumerate(t2[: mat_size]):
            v = -1
            if word1 == word2:
                v = 1
            matrix[i][j] = v
    return matrix


def process_cnn_match_pair(entity_type, matrix_size_1=10, matrix_size_2=5, shuffle=True, seed=42):
    file_dir = join(settings.DATA_DIR, entity_type)

    pairs_train = utils.load_json(file_dir, "{}_alignment_{}_pairs.json".format(entity_type, "train"))
    pairs_test = utils.load_json(file_dir, "{}_alignment_{}_pairs.json".format(entity_type, "test"))

    vocab, text_pipeline = build_tokenizer(entity_type, pairs_train, pairs_test)

    n_matrix = len(pairs_train)
    x_long = np.zeros((n_matrix, matrix_size_1, matrix_size_1))
    x_short = np.zeros((n_matrix, matrix_size_2, matrix_size_2))
    y = np.zeros(n_matrix, dtype=np.long)
    count = 0

    for i, pair in enumerate(pairs_train):
        if i % 100 == 0:
            print('pairs to matrices', i)
        cur_y = pair["label"]
        item1 = utils.normalize_text(pair["{}1".format(entity_type)])
        item2 = utils.normalize_text(pair["{}2".format(entity_type)])
        item1 = text_pipeline(item1)
        item2 = text_pipeline(item2)

        matrix1 = complete_sentence_pair_to_matrix(item1, item2, matrix_size_1)
        x_long[count] = utils.scale_matrix(matrix1)
        matrix2 = sentences_overlap_to_matrix(item1, item2, matrix_size_2)
        x_short[count] = utils.scale_matrix(matrix2)
        y[count] = cur_y
        count += 1

    n_test = len(pairs_test)
    x_test_long = np.zeros((n_test, matrix_size_1, matrix_size_1))
    x_test_short = np.zeros((n_test, matrix_size_2, matrix_size_2))
    y_test = np.zeros(n_test, dtype=np.long)
    count = 0
    for i, pair in enumerate(pairs_test):
        if i % 100 == 0:
            print('pairs to matrices', i)
        cur_y = pair["label"]
        item1 = utils.normalize_text(pair["{}1".format(entity_type)])
        item2 = utils.normalize_text(pair["{}2".format(entity_type)])
        item1 = text_pipeline(item1)
        item2 = text_pipeline(item2)

        matrix1 = complete_sentence_pair_to_matrix(item1, item2, matrix_size_1)
        x_test_long[count] = utils.scale_matrix(matrix1)
        matrix2 = sentences_overlap_to_matrix(item1, item2, matrix_size_2)
        x_test_short[count] = utils.scale_matrix(matrix2)
        y_test[count] = cur_y
        count += 1

    print("shuffle", shuffle)
    if shuffle:
        x_long, x_short, y = sklearn.utils.shuffle(
            x_long, x_short, y,
            random_state=seed
        )

    N = len(y)

    if entity_type == "aff":
        n_train = int(N*0.75)
        n_valid = int(N*0.25)
    elif entity_type == "venue":
        n_train = int(N / 5 * 4)
        n_valid = int(N / 5)
    else:
        raise NotImplementedError

    out_dir = join(settings.OUT_DIR, entity_type, "cnn")
    os.makedirs(out_dir, exist_ok=True)

    train_data = {}
    train_data["x1"] = x_long[:n_train]
    train_data["x2"] = x_short[:n_train]
    train_data["y"] = y[:n_train]
    print("train labels", len(train_data["y"]))

    valid_data = {}
    valid_data["x1"] = x_long[n_train:(n_train + n_valid)]
    valid_data["x2"] = x_short[n_train:(n_train + n_valid)]
    valid_data["y"] = y[n_train:(n_train + n_valid)]
    print("valid labels", len(valid_data["y"]), valid_data["y"])

    test_data = {}
    test_data["x1"] = x_test_long
    test_data["x2"] = x_test_short
    test_data["y"] = y_test
    print("test labels", len(test_data["y"]), test_data["y"])

    utils.dump_large_obj(train_data, out_dir, "{}_cnn_train.pkl".format(entity_type))
    utils.dump_large_obj(test_data, out_dir, "{}_cnn_test.pkl".format(entity_type))
    utils.dump_large_obj(valid_data, out_dir, "{}_cnn_valid.pkl".format(entity_type))


def process_cnn_author_match_pair(matrix_size_1=10, matrix_size_2=5, shuffle=True, seed=42):
    file_dir = join(settings.DATA_DIR, "author")

    pairs_train = utils.load_json(file_dir, "author_alignment_{}_pairs.json".format("train"))
    pairs_test = utils.load_json(file_dir, "author_alignment_{}_pairs.json".format("test"))
    labels_test = [x["label"] for x in pairs_test]

    pretrain_emb = np.load(join(file_dir, "large_cross_graph_node_emb.npy"))
    pretrain_emb = np.concatenate((pretrain_emb, np.zeros(shape=(2, 128))), axis=0)

    node_list = []
    with open(join(file_dir, "large_cross_graph_nodes_list.txt")) as rf:
        for i, line in enumerate(rf):
            node_list.append(line.strip())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    aminer_ego_author_attr = utils.load_json(file_dir, "aminer_ego_author_attr_dict.json")
    dblp_ego_author_attr = utils.load_json(file_dir, "dblp_ego_author_attr_dict.json")

    aminer_aid_to_vids = utils.load_json(file_dir, "aminer_aid_to_vids.json")
    dblp_aid_to_vids = utils.load_json(file_dir, "dblp_aid_to_vids.json")

    n_matrix = len(pairs_train)
    x_long = np.zeros((n_matrix, matrix_size_1, matrix_size_1))
    x_short = np.zeros((n_matrix, matrix_size_2, matrix_size_2))
    y = [x["label"] for x in pairs_train]
    count = 0

    for pair in tqdm(pairs_train):
        aid = pair["aminer"]
        did = pair["dblp"]
        pubs_a = aminer_ego_author_attr.get(aid, [])["pubs"][: matrix_size_1]
        pubs_d = dblp_ego_author_attr.get(did, [])["pubs"][: matrix_size_1]
        pubs_a_idx = [node_to_idx[x + "-pp"] for x in pubs_a]
        pubs_d_idx = [node_to_idx[x + "-pp"] for x in pubs_d]

        matrix1 = complete_sentence_pair_to_matrix(pubs_a_idx, pubs_d_idx, matrix_size_1, pretrain_emb)
        x_long[count] = utils.scale_matrix(matrix1)

        vids_a = aminer_aid_to_vids.get(aid, [])[: matrix_size_2]
        vids_d = dblp_aid_to_vids.get(aid, [])[: matrix_size_2]
        vids_a_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_a]
        vids_d_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_d]
        matrix2 = complete_sentence_pair_to_matrix(vids_a_idx, vids_d_idx, matrix_size_2, pretrain_emb)
        x_short[count] = utils.scale_matrix(matrix2)
        count += 1

    x_test_long = np.zeros((len(labels_test), matrix_size_1, matrix_size_1))
    x_test_short = np.zeros((len(labels_test), matrix_size_2, matrix_size_2))
    y_test = labels_test

    count = 0
    for pair in tqdm(pairs_test):
        aid = pair["aminer"]
        did = pair["dblp"]
        pubs_a = aminer_ego_author_attr.get(aid, [])["pubs"][: matrix_size_1]
        pubs_d = dblp_ego_author_attr.get(did, [])["pubs"][: matrix_size_1]
        pubs_a_idx = [node_to_idx[x + "-pp"] for x in pubs_a]
        pubs_d_idx = [node_to_idx[x + "-pp"] for x in pubs_d]

        matrix1 = complete_sentence_pair_to_matrix(pubs_a_idx, pubs_d_idx, matrix_size_1, pretrain_emb)
        x_test_long[count] = utils.scale_matrix(matrix1)

        vids_a = aminer_aid_to_vids.get(aid, [])[: matrix_size_2]
        vids_d = dblp_aid_to_vids.get(aid, [])[: matrix_size_2]
        vids_a_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_a]
        vids_d_idx = [node_to_idx[x["id"] + "-vv"] for x in vids_d]
        matrix2 = complete_sentence_pair_to_matrix(vids_a_idx, vids_d_idx, matrix_size_2, pretrain_emb)
        x_test_short[count] = utils.scale_matrix(matrix2)
        count += 1

    print("shuffle", shuffle)
    if shuffle:
        x_long, x_short, y = sklearn.utils.shuffle(
            x_long, x_short, y,
            random_state=seed
        )

    N = len(y)

    n_train = int(N / 9 * 8)
    n_valid = int(N / 9)

    out_dir = join(settings.OUT_DIR, "author", "cnn")
    os.makedirs(out_dir, exist_ok=True)

    train_data = {}
    train_data["x1"] = x_long[:n_train]
    train_data["x2"] = x_short[:n_train]
    train_data["y"] = y[:n_train]
    print("train labels", len(train_data["y"]))

    valid_data = {}
    valid_data["x1"] = x_long[n_train:(n_train + n_valid)]
    valid_data["x2"] = x_short[n_train:(n_train + n_valid)]
    valid_data["y"] = y[n_train:(n_train + n_valid)]
    print("valid labels", len(valid_data["y"]), valid_data["y"])

    test_data = {}
    test_data["x1"] = x_test_long
    test_data["x2"] = x_test_short
    test_data["y"] = y_test
    print("test labels", len(test_data["y"]), test_data["y"])

    utils.dump_large_obj(train_data, out_dir, "author_cnn_train.pkl")
    utils.dump_large_obj(test_data, out_dir, "author_cnn_test.pkl")
    utils.dump_large_obj(valid_data, out_dir, "author_cnn_valid.pkl")


def gen_author_paired_subgraphs():
    author_dir = join(settings.DATA_DIR, "author")

    pairs_train = utils.load_json(author_dir, "author_alignment_{}_pairs.json".format("train"))
    pairs_test = utils.load_json(author_dir, "author_alignment_{}_pairs.json".format("test"))
    labels_train = [x["label"] for x in pairs_train]
    labels_test = [x["label"] for x in pairs_test]

    adjs = []
    vertex_ids = []
    vertex_types = []

    adjs_test = []
    vertex_ids_test = []
    vertex_types_test = []
    n_nodes_ego = 382

    aminer_ego_author_attr = utils.load_json(author_dir, "aminer_ego_author_attr_dict.json")
    aminer_aid_to_sorted_pubs = utils.load_json(author_dir, "aminer_aid_to_sorted_pubs.json")

    dblp_ego_author_attr = utils.load_json(author_dir, "dblp_ego_author_attr_dict.json")
    dblp_aid_to_sorted_pids = utils.load_json(author_dir, "dblp_aid_to_sorted_pubs.json")

    aminer_aid_to_vids = utils.load_json(author_dir, "aminer_aid_to_vids.json")
    dblp_aid_to_vids = utils.load_json(author_dir, "dblp_aid_to_vids.json")

    aminer_aid_to_coauthors = utils.load_json(author_dir, "aminer_aid_to_coauthors.json")
    dblp_aid_to_coauthors = utils.load_json(author_dir, "dblp_aid_to_coauthors.json")

    coauthor_dict_cache = {**aminer_aid_to_coauthors, **dblp_aid_to_coauthors}
    ego_attr = {**aminer_ego_author_attr, **dblp_ego_author_attr}
    aid_to_pids_ego = {x: ego_attr[x].get("pubs", []) for x in ego_attr}
    aid_to_pids = {**aid_to_pids_ego, **aminer_aid_to_sorted_pubs, **dblp_aid_to_sorted_pids}
    aid_to_vids = {**aminer_aid_to_vids, **dblp_aid_to_vids}

    for i, pair in enumerate(pairs_train + pairs_test):
        if i < len(labels_train):
            cur_adjs = adjs
            cur_vertex_ids = vertex_ids
            cur_vertex_types = vertex_types
        else:
            cur_adjs = adjs_test
            cur_vertex_ids = vertex_ids_test
            cur_vertex_types = vertex_types_test

        node_set = set()
        node_list = []
        node_type_list = []
        n_authors = 0
        n_venues = 0
        n_papers = 0
        aaid, daid = pair['aminer'], pair['dblp']
        aaid_dec = '{}-aa'.format(aaid)
        daid_dec = '{}-ad'.format(daid)
        n_authors += 2

        # focal_node1 = aaid
        # focal_node2 = daid
        node_set.update({aaid, daid})
        node_list += [aaid_dec, daid_dec]
        node_type_list += [settings.AUTHOR_TYPE, settings.AUTHOR_TYPE]

        # 1-ego venues
        a_venues = aminer_aid_to_vids.get(aaid, [])[:10]
        d_venues = dblp_aid_to_vids.get(daid, [])[:10]
        venues_1_ego = [v['id'] for v in a_venues + d_venues]
        for v in venues_1_ego:
            if v not in node_set:
                node_set.add(v)
                v_dec = '{}-vv'.format(v)
                n_venues += 1
                node_list.append(v_dec)
                node_type_list.append(settings.VENUE_TYPE)

        # 1-ego papers
        a_pubs = aminer_ego_author_attr.get(aaid, {}).get('pubs', [])[:20]
        d_pubs = dblp_ego_author_attr.get(daid, {}).get('pubs', [])[:20]
        pubs_1_ego = {item for item in a_pubs + d_pubs if item not in node_set}
        node_set.update(pubs_1_ego)
        node_list += ['{}-pp'.format(item) for item in pubs_1_ego]
        node_type_list += [settings.PAPER_TYPE] * len(pubs_1_ego)
        n_papers += len(pubs_1_ego)

        # 1-ego coauthors
        a_coauthors = aminer_aid_to_coauthors.get(aaid, [])[:10]
        for cur_aid in a_coauthors:
            if cur_aid not in node_set:
                node_set.add(cur_aid)
                node_list.append('{}-aa'.format(cur_aid))
                node_type_list.append(settings.AUTHOR_TYPE)
                n_authors += 1

        d_coauthors = dblp_aid_to_coauthors.get(daid, [])[:10]
        d_coauthors = {a for a in d_coauthors if a not in node_set}
        node_set.update(d_coauthors)
        node_list += ['{}-ad'.format(a) for a in d_coauthors]
        node_type_list += [settings.AUTHOR_TYPE] * len(d_coauthors)
        n_authors += len(d_coauthors)

        # 2-ego
        for a in a_coauthors:
            cur_venues = aminer_aid_to_vids.get(a, [])[:5]
            cur_vids = {v['id'] for v in cur_venues if v['id'] not in node_set}
            node_set.update(cur_vids)
            node_list += ['{}-vv'.format(v) for v in cur_vids]
            node_type_list += [settings.VENUE_TYPE] * len(cur_vids)
            n_venues += len(cur_vids)

        for a in d_coauthors:
            cur_venues = dblp_aid_to_vids.get(a, [])[:5]
            cur_vids = {v['id'] for v in cur_venues if v['id'] not in node_set}
            node_set.update(cur_vids)
            node_list += ['{}-vv'.format(v) for v in cur_vids]
            node_type_list += [settings.VENUE_TYPE] * len(cur_vids)
            n_venues += len(cur_vids)

        a_pubs_2 = []
        for a in a_coauthors:
            cur_pubs = aminer_aid_to_sorted_pubs.get(a, [])[:10]
            a_pubs_2 += cur_pubs

        d_pubs_2 = []
        for a in d_coauthors:
            cur_pubs = dblp_aid_to_sorted_pids.get(a, [])[:10]
            d_pubs_2 += cur_pubs

        pubs_2_ego = {item for item in a_pubs_2 + d_pubs_2 if item not in node_set}
        node_set.update(pubs_2_ego)
        node_list += ['{}-pp'.format(p) for p in pubs_2_ego]
        node_type_list += [settings.PAPER_TYPE] * len(pubs_2_ego)
        n_papers += len(pubs_2_ego)

        cur_n_nodes_real = len(node_list)
        assert len(node_set) == len(node_list) == len(node_type_list)
        assert n_authors <= 22 and n_venues <= 120 and n_papers <= 240

        # padding
        for v_idx in range(len(node_set), n_nodes_ego):
            node_list.append('-1')
            node_type_list.append(settings.AUTHOR_TYPE)
        assert len(node_list) == n_nodes_ego

        cur_vertex_ids.append(node_list)
        cur_vertex_types.append(node_type_list)

        node_to_idx = {eid[:-3]: i for i, eid in enumerate(node_list)}
        adj = np.zeros((n_nodes_ego, n_nodes_ego), dtype=np.bool_)
        nnz = 0
        for ii in range(cur_n_nodes_real):
            v1 = node_list[ii][:-3]
            t1 = node_type_list[ii]
            if t1 == settings.AUTHOR_TYPE:
                v_nbrs = aid_to_vids.get(v1, [])
                v_nbrs = {item['id'] for item in v_nbrs}
                v_nbrs_filtered = {item for item in v_nbrs if item in node_set}
                nbrs_set = v_nbrs_filtered

                p_nbrs = aid_to_pids.get(v1, [])
                p_nbrs_filtered = {item for item in p_nbrs if item in node_set}
                nbrs_set.update(p_nbrs_filtered)

                a_nbrs = coauthor_dict_cache.get(v1, [])
                a_nbrs_filtered = {item for item in a_nbrs if item in node_set}
                nbrs_set.update(a_nbrs_filtered)

                for nbr in nbrs_set:
                    nbr_idx_map = node_to_idx[nbr]
                    adj[ii, nbr_idx_map] = True
                    adj[nbr_idx_map, ii] = True
                    nnz += 1

        cur_adjs.append(adj)

        if i % 100 == 0:
            logger.info('***********iter %d nnz %d', i, nnz)
            logger.info('n_nodes real %d, n_authors %d, n_venues %d, n_papers %d.',
                        cur_n_nodes_real, n_authors, n_venues, n_papers)

    out_dir = join(settings.OUT_DIR, "author", "hgat")
    os.makedirs(out_dir, exist_ok=True)
    np.save(join(out_dir, 'vertex_id_train.npy'), np.array(vertex_ids))
    np.save(join(out_dir, 'vertex_types_train.npy'), np.array(vertex_types))
    np.save(join(out_dir, 'adjacency_matrix_train.npy'), np.array(adjs))
    np.save(join(out_dir, 'label_train.npy'), np.array(labels_train))

    np.save(join(out_dir, 'vertex_id_test.npy'), np.array(vertex_ids_test))
    np.save(join(out_dir, 'vertex_types_test.npy'), np.array(vertex_types_test))
    np.save(join(out_dir, 'adjacency_matrix_test.npy'), np.array(adjs_test))
    np.save(join(out_dir, 'label_test.npy'), np.array(labels_test))


if __name__ == "__main__":
    # for svm
    # gen_short_texts_sim_stat_features(entity_type="aff")
    # gen_short_texts_sim_stat_features(entity_type="venue")
    # gen_author_sim_struct_features()

    # for rnn-based matching
    # process_rnn_match_pair(entity_type="aff", max_seq1_len=10, max_seq2_len=5)
    # process_rnn_match_pair(entity_type="venue", max_seq1_len=10, max_seq2_len=5)
    # process_rnn_author_match_pair(max_seq1_len=10, max_seq2_len=5)

    # for cnn-based matching
    # process_cnn_match_pair(entity_type="aff")
    # process_cnn_match_pair(entity_type="venue")
    # process_cnn_author_match_pair()

    # for hgat-based matching
    gen_author_paired_subgraphs()
    logger.info("done")
