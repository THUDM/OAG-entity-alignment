import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, n_type_nodes, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.n_type_nodes = n_type_nodes
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, self.n_type_nodes))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, self.n_type_nodes))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj, v_types):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        v_types = v_types.unsqueeze(1)
        v_types = v_types.expand(-1, self.n_head, -1, -1)
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_src = torch.sum(torch.mul(attn_src, v_types), dim=3, keepdim=True)
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn_dst = torch.sum(torch.mul(attn_dst, v_types), dim=3, keepdim=True)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1)  # bs x 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MatchBatchHGAT(nn.Module):
    def __init__(self, n_type_nodes, n_units=[1433, 8, 7], n_head=8, dropout=0.1,
                 attn_dropout=0.0, instance_normalization=False):
        super(MatchBatchHGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization

        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)

        d_hidden = n_units[-1]

        self.fc1 = torch.nn.Linear(d_hidden * 2, d_hidden * 3)
        self.fc2 = torch.nn.Linear(d_hidden * 3, d_hidden)
        self.fc3 = torch.nn.Linear(d_hidden, 2)

        self.attentions = BatchMultiHeadGraphAttention(n_head=n_head,
                                                       f_in=n_units[0],
                                                       f_out=n_units[1],
                                                       attn_dropout=attn_dropout,
                                                       n_type_nodes=n_type_nodes)

        self.out_att = BatchMultiHeadGraphAttention(n_head=1,
                                                    f_in=n_head * n_units[1],
                                                    f_out=n_units[2],
                                                    attn_dropout=attn_dropout,
                                                    n_type_nodes=n_type_nodes)

    def forward(self, emb, adj, v_types, x_stat):
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        bs, n = adj.size()[:2]
        x = self.attentions(emb, adj, v_types)
        x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj, v_types)
        x = F.elu(x)
        x = x.squeeze()

        left_hidden = x[:, 0, :]
        right_hidden = x[:, 1, :]
        v_sim_mul = torch.mul(left_hidden, right_hidden)

        x_stat = torch.cat((x_stat, x_stat, x_stat, x_stat), dim=1)
        v_sim_mul = torch.cat((v_sim_mul, x_stat), dim=1)

        v_sim = self.fc1(v_sim_mul)
        v_sim = F.relu(v_sim)
        v_sim = self.fc2(v_sim)
        v_sim = F.relu(v_sim)
        scores = self.fc3(v_sim)
        return F.log_softmax(scores, dim=1)
