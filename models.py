import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, pretrain_emb, embedding_size=128, hidden_size=32, dropout=0.2,
                 multiple=0, use_seq_num=2):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size

        # embedding layer
        self.embed_seq = nn.Embedding(self.vocab_size + 1, embedding_size)
        self.embed_keyword_seq = nn.Embedding(self.vocab_size + 1, embedding_size)
        self.embed_seq.weight = nn.Parameter(pretrain_emb, requires_grad=False)
        self.embed_keyword_seq.weight = nn.Parameter(pretrain_emb, requires_grad=False)

        self.embed_seq.weight.requires_grad = False
        self.embed_keyword_seq.weight.requires_grad = False

        # LSTM layer
        self.lstm_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.lstm_seq2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        self.lstm_key_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.lstm_key_seq2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        self.output = nn.Sequential(
            nn.Linear(6 * hidden_size + 3 * multiple, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.output_for_one_seq = nn.Sequential(
            nn.Linear(3 * hidden_size + 3 * multiple, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.use_seq_num = use_seq_num
        self.output_final = nn.Linear(16, 2)

    def forward(self, x1, x2, x1_keywords, x2_keywords):
        x1 = self.embed_seq(x1)
        x2 = self.embed_seq(x2)
        x1_keywords = self.embed_keyword_seq(x1_keywords)
        x2_keywords = self.embed_keyword_seq(x2_keywords)

        x1, _ = self.lstm_seq1(x1)
        x1, _ = self.lstm_seq2(x1)
        x2, _ = self.lstm_seq1(x2)
        x2, _ = self.lstm_seq2(x2)
        x1_keywords, _ = self.lstm_key_seq1(x1_keywords)
        x1_keywords, _ = self.lstm_key_seq2(x1_keywords)
        x2_keywords, _ = self.lstm_key_seq1(x2_keywords)
        x2_keywords, _ = self.lstm_key_seq2(x2_keywords)
        minus = x1_keywords[:, -1, :] - x2_keywords[:, -1, :]
        minus_key = x1[:, -1, :] - x2[:, -1, :]

        if self.use_seq_num == 2:
            concat_input = torch.cat(
                (minus,
                 minus_key,
                 x1[:, -1, :],
                 x2[:, -1, :],
                 x1_keywords[:, -1, :],
                 x2_keywords[:, -1, :],
                 ), dim=1)
            output_hidden = self.output(concat_input)
        else:
            concat_input = torch.cat(
                (minus_key,
                 x1[:, -1, :],
                 x2[:, -1, :],
                 ), dim=1)
            output_hidden = self.output_for_one_seq(concat_input)

        output = self.output_final(output_hidden)

        return torch.log_softmax(output, dim=1)
