import os
import json
import torch
from torch import nn
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from models import CANN, GCN
from curvature import node_curvature_from_edges
from torch.utils.data import Subset

class GraphModel(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, use_cann=False):
        super().__init__()
        self.use_cann = use_cann
        if use_cann:
            self.enc = CANN(in_dim, hidden, hidden, layers=2)
        else:
            self.enc = GCN(in_dim, hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data, K=None):
        if self.use_cann:
            h = self.enc(data.x.float(), data.edge_index, K)
        else:
            h = self.enc(data.x.float(), data.edge_index)
        hg = global_mean_pool(h, data.batch)
        return self.lin(hg)

def run_zinc(root, epochs=20, hidden=64):
    ds = ZINC(root=root, subset=True)
    n = len(ds)
    if n >= 12000:
        idx_train = list(range(0, 10000))
        idx_val = list(range(10000, 11000))
        idx_test = list(range(11000, 12000))
    else:
        t = int(n * 0.8)
        v = int(n * 0.1)
        idx_train = list(range(0, t))
        idx_val = list(range(t, t + v))
        idx_test = list(range(t + v, n))
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(ds, idx_val), batch_size=128)
    test_loader = DataLoader(Subset(ds, idx_test), batch_size=128)
    gcn = GraphModel(in_dim=ds.num_features, hidden=hidden//2, out_dim=1, use_cann=False)
    cann = GraphModel(in_dim=ds.num_features, hidden=hidden, out_dim=1, use_cann=True)
    opt1 = torch.optim.Adam(gcn.parameters(), lr=0.001)
    opt2 = torch.optim.Adam(cann.parameters(), lr=0.001)
    l1 = nn.L1Loss()
    def eval_metrics(model, loader, use_cann):
        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for batch in loader:
                if use_cann:
                    K = torch.tensor(node_curvature_from_edges(batch.num_nodes, batch.edge_index.t().tolist(), approx=True, sample_neighbors=8, sample_pairs=8), dtype=torch.float32)
                    out = model(batch, K)
                else:
                    out = model(batch)
                ys.append(batch.y.view(-1).float())
                ps.append(out.view(-1).float())
        y = torch.cat(ys)
        p = torch.cat(ps)
        mae = torch.mean(torch.abs(p - y)).item()
        rmse = torch.sqrt(torch.mean((p - y) ** 2)).item()
        ry = torch.argsort(torch.argsort(y))
        rp = torch.argsort(torch.argsort(p))
        n = float(y.numel())
        d = (ry.float() - rp.float()) ** 2
        spearman = 1.0 - (6.0 * torch.sum(d).item()) / (n * (n * n - 1.0))
        return mae, rmse, spearman
    def eval_robust_delta(model, loader, use_cann, drop_ratio=0.1):
        model.eval()
        base = []
        pert = []
        with torch.no_grad():
            for batch in loader:
                if use_cann:
                    K = torch.tensor(node_curvature_from_edges(batch.num_nodes, batch.edge_index.t().tolist(), approx=True, sample_neighbors=8, sample_pairs=8), dtype=torch.float32)
                    out = model(batch, K).view(-1).float()
                else:
                    out = model(batch).view(-1).float()
                base.append(torch.mean(torch.abs(out - batch.y.view(-1).float())).item())
                ei = batch.edge_index
                m = ei.size(1)
                keep = int(m * (1.0 - drop_ratio))
                idx = torch.randperm(m)[:keep]
                batch_pert = type(batch)()
                batch_pert.x = batch.x
                batch_pert.edge_index = ei[:, idx]
                batch_pert.batch = batch.batch
                batch_pert.y = batch.y
                if use_cann:
                    Kp = torch.tensor(node_curvature_from_edges(batch.num_nodes, batch_pert.edge_index.t().tolist(), approx=True, sample_neighbors=8, sample_pairs=8), dtype=torch.float32)
                    outp = model(batch_pert, Kp).view(-1).float()
                else:
                    outp = model(batch_pert).view(-1).float()
                pert.append(torch.mean(torch.abs(outp - batch.y.view(-1).float())).item())
        return (sum(pert) / max(1, len(pert))) - (sum(base) / max(1, len(base)))
    for _ in range(epochs):
        gcn.train()
        for batch in train_loader:
            opt1.zero_grad()
            out = gcn(batch)
            loss = l1(out.squeeze(), batch.y.squeeze())
            loss.backward()
            opt1.step()
        cann.train()
        for batch in train_loader:
            opt2.zero_grad()
            K = torch.tensor(node_curvature_from_edges(batch.num_nodes, batch.edge_index.t().tolist(), approx=True, sample_neighbors=8, sample_pairs=8), dtype=torch.float32)
            out = cann(batch, K)
            loss = l1(out.squeeze(), batch.y.squeeze())
            loss.backward()
            opt2.step()
    g_mae, g_rmse, g_sp = eval_metrics(gcn, test_loader, False)
    c_mae, c_rmse, c_sp = eval_metrics(cann, test_loader, True)
    g_delta = eval_robust_delta(gcn, test_loader, False, drop_ratio=0.1)
    c_delta = eval_robust_delta(cann, test_loader, True, drop_ratio=0.1)
    return {
        "GCN": {"MAE": round(g_mae, 3), "RMSE": round(g_rmse, 3), "Spearman": round(g_sp, 3), "RobustDeltaMAE": round(g_delta, 3)},
        "CANN": {"MAE": round(c_mae, 3), "RMSE": round(c_rmse, 3), "Spearman": round(c_sp, 3), "RobustDeltaMAE": round(c_delta, 3)}
    }

def main():
    root = os.path.join("data", "real", "molecular", "zinc")
    out_path = os.path.abspath(os.path.join("data", "results_graph.json"))
    res = {"ZINC": run_zinc(root)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
