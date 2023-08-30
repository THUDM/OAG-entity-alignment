import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        if 'glm' in lm:
            return AutoTokenizer.from_pretrained(lm,trust_remote_code=True)
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size
        self.lm=lm

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]
        
        # left + right
        # if 'glm' in self.lm:
        #     x = self.tokenizer(text=left,
        #                        text_pair=right,
        #                        max_length=self.max_len,
        #                        padding='max_length',
        #                        return_tensors='pt',
        #                        truncation=True)
        #     for k,v in x.items():
        #         x[k]=v.squeeze(0)
        # else:
        x = self.tokenizer.encode(text=left,
                                text_pair=right,
                                max_length=self.max_len,
                                truncation=True)

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            # if 'glm' in self.lm:
            #     x_aug = self.tokenizer(text=left,
            #                           text_pair=right,
            #                           max_length=self.max_len,
            #                           padding='max_length',
            #                           return_tensors='pt',
            #                           truncation=True)
            #     for k,v in x_aug.items():
            #         x_aug[k]=v.squeeze(0)
            # else:
            x_aug = self.tokenizer.encode(text=left,
                                    text_pair=right,
                                    max_length=self.max_len,
                                    truncation=True)
            return x, x_aug, self.labels[idx]
        else:
            return x, self.labels[idx]


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)

    @staticmethod
    def glmpad(batch):
        if len(batch[0])==3:
            x1, x2, y = zip(*batch)
            
            
            maxlen = max([len(x) for x in x1+x2])
            
            attention_mask1=[[1]*len(xi)+[0]*(maxlen - len(xi)) for xi in x1]
            x1 = [xi + [2]*(maxlen - len(xi)) for xi in x1]
            
            attention_mask2=[[1]*len(xi)+[0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [2]*(maxlen - len(xi)) for xi in x2]
            
            return {'input_ids':torch.LongTensor(x1),'attention_mask':torch.LongTensor(attention_mask1)}, \
                   {'input_ids':torch.LongTensor(x2),'attention_mask':torch.LongTensor(attention_mask2)}, \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            
            attention_mask=[[1]*len(xi)+[0]*(maxlen - len(xi)) for xi in x12]
            x12 = [xi + [2]*(maxlen - len(xi)) for xi in x12]
            return {'input_ids':torch.LongTensor(x12),'attention_mask':torch.LongTensor(attention_mask)}, \
                   torch.LongTensor(y)