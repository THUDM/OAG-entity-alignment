from os.path import join
import json
import sklearn
from tqdm import tqdm
from collections import Counter

import utils


def calc_keywords_seqs(x1, x2):
    N = len(x1)
    x1_keywords = []
    x2_keywords = []
    for i in tqdm(range(N)):
        item1 = x1[i].split()
        item2 = x2[i].split()
        overlap = Counter(item1) & Counter(item2)
        item1_new = []
        item2_new = []
        for w in item1:
            if w in overlap:
                item1_new.append(w)
        for w in item2:
            if w in overlap:
                item2_new.append(w)
        x1_keywords.append(" ".join(item1_new))
        x2_keywords.append(" ".join(item2_new))
    return x1_keywords, x2_keywords


def process_entity_pairs(entity_type="aff"):
    data_dir = "data/oag/{}/".format(entity_type)
    file_name = entity_type + "_alignment_{}_pairs.json"
    
    with open(join(data_dir, file_name.format("train"))) as rf:
        data = json.load(rf)
    
    with open(join(data_dir, file_name.format("test"))) as rf:
        data_test = json.load(rf)    

    N = len(data)

    data = sklearn.utils.shuffle(
        data,
        random_state=42
    )

    if entity_type == "aff":
        n_train = int(N*0.75)
        n_valid = int(N*0.25)
    elif entity_type == "venue":
        n_train = int(N / 5 * 4)
        n_valid = int(N / 5)
    else:
        raise NotImplementedError

    all_data = [data[:n_train], data[n_train: n_train+n_valid], data_test]
    assert len(all_data) == 3
    roles = ["train", "valid", "test"]

    for i in range(3):
        cur_data = all_data[i]
        role = roles[i]
        x1 = []
        x2 = []
        labels = []
        for pair in cur_data:
            item1 = utils.normalize_text(pair["{}1".format(entity_type)])
            item2 = utils.normalize_text(pair["{}2".format(entity_type)])
            x1.append(item1)
            x2.append(item2)
            labels.append(pair["label"])

        x1_keywords, x2_keywords = calc_keywords_seqs(x1, x2)
        cur_wf = open(join(data_dir, "{}.txt".format(role)), "w")
        cur_len = len(labels)

        for j in range(cur_len):
            cur_wf.write("COL name VAL ")
            cur_wf.write(x1[j].replace("\t", " ").replace("\n", " ").strip() + " ")
            cur_wf.write("COL keyword VAL ")
            cur_wf.write(x1_keywords[j].replace("\t", " ").replace("\n", " ").strip() + "\t")

            cur_wf.write("COL name VAL ")
            cur_wf.write(x2[j].replace("\t", " ").replace("\n", " ").strip() + " ")
            cur_wf.write("COL keyword VAL ")
            cur_wf.write(x2_keywords[j].replace("\t", " ").replace("\n", " ").strip() + "\t")

            cur_wf.write(str(labels[j]) + "\n")
            cur_wf.flush()

        cur_wf.close()


if __name__ == "__main__":
    process_entity_pairs(entity_type="aff")
