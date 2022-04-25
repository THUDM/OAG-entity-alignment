import torch
from torch import nn
import torch.nn.functional as F


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

        self.lstm_key_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout,
                                     batch_first=True)
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


class CNNMatchModel(nn.Module):
    def __init__(self, input_matrix_size1, input_matrix_size2, mat1_channel1, mat1_kernel_size1,
                 mat1_channel2, mat1_kernel_size2, mat2_channel1, mat2_kernel_size1, hidden1,
                 hidden2):
        super(CNNMatchModel, self).__init__()
        self.mat_size1 = input_matrix_size1
        self.mat_size2 = input_matrix_size2

        self.conv1_1 = nn.Conv2d(1, mat1_channel1,
                                 mat1_kernel_size1)  # n*mat1_channel1*(input_matrix_size1-mat1_kernel_size1+1)*(input_matrix_size1-mat1_kernel_size1+1)
        self.conv1_2 = nn.Conv2d(mat1_channel1, mat1_channel2,
                                 mat1_kernel_size2)  # n*mat1_channel2*(input_matrix_size1-mat1_kernel_size1-mat1_kernel_size2+2)*(input_matrix_size1-mat1_kernel_size1-mat1_kernel_size2+2)
        self.mat1_flatten_dim = mat1_channel2 * ((input_matrix_size1 - mat1_kernel_size1 - mat1_kernel_size2 + 2) ** 2)

        self.conv2_1 = nn.Conv2d(1, mat2_channel1,
                                 mat2_kernel_size1)  # n*mat2_channel1*(input_matrix_size2-mat2_kernel_size1+1)*(input_matrix_size2-mat2_kernel_size1+1)
        self.mat2_flatten_dim = mat2_channel1 * ((input_matrix_size2 - mat2_kernel_size1 + 1) ** 2)
        print("flat cnn", self.mat1_flatten_dim, self.mat2_flatten_dim)

        self.fc_out = nn.Sequential(
            nn.Linear(self.mat1_flatten_dim + self.mat2_flatten_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, batch_matrix1, batch_matrix2):
        batch_matrix1 = batch_matrix1.unsqueeze(1)
        batch_matrix2 = batch_matrix2.unsqueeze(1)

        mat1 = F.relu(self.conv1_1(batch_matrix1))
        mat1 = F.relu(self.conv1_2(mat1))
        mat1 = mat1.view(-1, self.mat1_flatten_dim)

        mat2 = F.relu(self.conv2_1(batch_matrix2))
        mat2 = mat2.view(-1, self.mat2_flatten_dim)

        hidden = torch.cat((mat1, mat2), 1)
        out = self.fc_out(hidden)

        return F.log_softmax(out, dim=1)
