from os.path import join
import json
import unicodedata
import pickle
from torch.utils.data.sampler import Sampler

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

stopwords = {'at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
             'is', 'are', 'can'}


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def dump_large_obj(obj, wfdir, wfname):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    with open(join(wfdir, wfname), 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def load_large_obj(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'rb') as rf:
        obj = pickle.load(rf)
        logger.info('%s loaded', rfname)
        return obj


def remove_marks(txt):
    """This method removes all diacritic marks from the given string"""
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)


def normalize_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = remove_marks(text)
    text = " ".join([x for x in text.split() if x not in stopwords])
    return text


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired data points
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
