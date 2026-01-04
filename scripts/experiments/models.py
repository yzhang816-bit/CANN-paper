import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.c1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.c2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=2, dropout=0.5):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=heads, concat=False, dropout=dropout)
        self.c2 = GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.c1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.c2(x, edge_index)
        return x

class CANNLayer(MessagePassing):
    def __init__(self, dim, heads=4):
        super().__init__(aggr='add')
        self.heads = heads
        self.Wk = nn.Linear(1, dim)
        self.Wq = nn.Linear(dim, dim * heads, bias=False)
        self.Wv = nn.Linear(dim, dim * heads, bias=False)
        self.a = nn.Parameter(torch.zeros(heads, 3 * dim))
        self.W = nn.Linear(dim * heads, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))
        self.ln = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x, edge_index, K):
        ku = self.Wk(K.unsqueeze(1))
        q = self.Wq(x)
        v = self.Wv(x)
        out = self.propagate(edge_index, x=x, q=q, v=v, ku=ku)
        out = self.W(out)
        out = out + x
        out = self.ln(out)
        g = torch.sigmoid(self.gate(ku))
        out = torch.relu(out * (1 + g) + self.b)
        return out

    def message(self, x_j, q_i, v_j, ku_i, index, ptr, size_i):
        h = q_i.size(-1) // self.heads
        qi = q_i.view(q_i.size(0), self.heads, h)
        vj = v_j.view(v_j.size(0), self.heads, h)
        ku = ku_i.unsqueeze(1).expand(-1, self.heads, h)
        c = torch.cat([ku, qi, vj], dim=-1)
        a = (self.a.unsqueeze(0) * c).sum(-1)
        e = torch.nn.functional.leaky_relu(a)
        alpha = softmax(e, index)
        m = alpha.unsqueeze(-1) * vj
        m = m.view(m.size(0), self.heads * h)
        return m

class CANN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([CANNLayer(hidden, heads=heads) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, K):
        x = torch.relu(self.input(x))
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, K)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return self.out(x)

class CANN_Simple(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_dim + 1, hidden, heads=heads, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, K):
        x_aug = torch.cat([x, K.view(-1, 1)], dim=1)
        x = self.conv1(x_aug, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
