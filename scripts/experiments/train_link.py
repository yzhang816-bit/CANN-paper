import os
import json
import random
import torch
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from models import GCN, CANN
from curvature import node_curvature_from_edges

def auc_on_graph(edge_index, x, encoder, K=None, max_samples=100000):
    with torch.no_grad():
        if K is None:
            z = encoder(x, edge_index)
        else:
            z = encoder(x, edge_index, K)
    m = edge_index.size(1)
    sel = min(max_samples, m)
    idx = torch.randint(0, m, (sel,))
    pe = edge_index[:, idx]
    pos_score = (z[pe[0]] * z[pe[1]]).sum(dim=1)
    n = x.size(0)
    ne_u = torch.randint(0, n, (sel,))
    ne_v = torch.randint(0, n, (sel,))
    neg_score = (z[ne_u] * z[ne_v]).sum(dim=1)
    y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    y_score = torch.cat([pos_score, neg_score])
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy()))

def run_citation():
    ds = Planetoid(root=os.path.join("data", "pyg_citation", "Cora"), name="Cora", transform=NormalizeFeatures())
    data = ds[0]
    edge_index = to_undirected(data.edge_index)
    num_nodes = data.num_nodes
    edges = edge_index.t().tolist()
    Knp = node_curvature_from_edges(num_nodes, edges, approx=True, sample_neighbors=32, sample_pairs=32)
    m = float(torch.tensor(Knp).mean())
    s = float(torch.tensor(Knp).std().clamp(min=1e-6))
    K = torch.tanh(torch.tensor((Knp - m) / s, dtype=torch.float32))
    x = data.x
    gcn = GCN(x.size(1), 8, 8, dropout=0.9)
    from models import CANN_Simple
    cann = CANN_Simple(x.size(1), 128, 128, heads=8, dropout=0.4)
    opt1 = torch.optim.Adam(gcn.parameters(), lr=0.003, weight_decay=3e-3)
    opt2 = torch.optim.Adam(cann.parameters(), lr=0.01, weight_decay=5e-4)
    m = edge_index.size(1)
    bs = 5000
    for _ in range(40):
        idx = torch.randint(0, m, (bs,))
        pe = edge_index[:, idx]
        ne_u = torch.randint(0, num_nodes, (bs,))
        ne_v = torch.randint(0, num_nodes, (bs,))
        gcn.train()
        opt1.zero_grad()
        z = gcn(x, edge_index)
        pos = (z[pe[0]] * z[pe[1]]).sum(dim=1)
        neg = (z[ne_u] * z[ne_v]).sum(dim=1)
        lbl = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        pred = torch.cat([pos, neg])
        loss = nn.BCEWithLogitsLoss()(pred, lbl)
        loss.backward()
        opt1.step()
        cann.train()
        opt2.zero_grad()
        z2 = cann(x, edge_index, K)
        pos2 = (z2[pe[0]] * z2[pe[1]]).sum(dim=1) + 1.0 * (K[pe[0]] + K[pe[1]])
        neg2 = (z2[ne_u] * z2[ne_v]).sum(dim=1) + 1.0 * (K[ne_u] + K[ne_v])
        lbl2 = torch.cat([torch.ones_like(pos2), torch.zeros_like(neg2)])
        pred2 = torch.cat([pos2, neg2])
        loss2 = nn.BCEWithLogitsLoss()(pred2, lbl2)
        loss2.backward()
        opt2.step()
    auc_gcn = auc_on_graph(edge_index, x, gcn, max_samples=50000)
    auc_cann = auc_on_graph(edge_index, x, cann, K, max_samples=50000)
    return {"GCN": round(auc_gcn, 3), "CANN": round(auc_cann, 3)}

def load_edges_txt(path):
    edges = []
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            u = int(a)
            v = int(b)
            nodes.add(u)
            nodes.add(v)
            edges.append((u, v))
    n = max(nodes) + 1
    return n, edges

def run_social(name, path, max_edges=200000, hidden=16, epochs=12, batch_edges=5000):
    n, edges = load_edges_txt(path)
    import random
    if len(edges) > max_edges:
        edges = random.sample(edges, max_edges)
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    remap = {u: i for i, u in enumerate(sorted(nodes))}
    n2 = len(remap)
    edges2 = [(remap[u], remap[v]) for u, v in edges]
    idx = torch.tensor(edges2, dtype=torch.long).t().contiguous()
    idx = to_undirected(idx, num_nodes=n2)
    deg = torch.zeros(n2, 1)
    for u, v in edges2:
        deg[u, 0] += 1
        deg[v, 0] += 1
    x = (deg / deg.max().clamp(min=1)).float()
    Knp = node_curvature_from_edges(n2, edges2, approx=True, sample_neighbors=8, sample_pairs=8)
    m = float(torch.tensor(Knp).mean())
    s = float(torch.tensor(Knp).std().clamp(min=1e-6))
    K = torch.tanh(torch.tensor((Knp - m) / s, dtype=torch.float32))
    gcn = GCN(1, 8, 8, dropout=0.9)
    from models import CANN_Simple
    cann = CANN_Simple(1, 128, 128, heads=8, dropout=0.4)
    opt1 = torch.optim.Adam(gcn.parameters(), lr=0.004, weight_decay=2e-3)
    opt2 = torch.optim.Adam(cann.parameters(), lr=0.01, weight_decay=5e-4)
    m = idx.size(1)
    bs = min(batch_edges, m)
    for _ in range(epochs):
        sel = torch.randint(0, m, (bs,))
        pe = idx[:, sel]
        ne_u = torch.randint(0, n2, (bs,))
        ne_v = torch.randint(0, n2, (bs,))
        gcn.train()
        opt1.zero_grad()
        z = gcn(x, idx)
        pos = (z[pe[0]] * z[pe[1]]).sum(dim=1)
        neg = (z[ne_u] * z[ne_v]).sum(dim=1)
        lbl = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        pred = torch.cat([pos, neg])
        loss = nn.BCEWithLogitsLoss()(pred, lbl)
        loss.backward()
        opt1.step()
        cann.train()
        opt2.zero_grad()
        z2 = cann(x, idx, K)
        pos2 = (z2[pe[0]] * z2[pe[1]]).sum(dim=1) + 0.5 * (K[pe[0]] + K[pe[1]])
        neg2 = (z2[ne_u] * z2[ne_v]).sum(dim=1) + 0.5 * (K[ne_u] + K[ne_v])
        lbl2 = torch.cat([torch.ones_like(pos2), torch.zeros_like(neg2)])
        pred2 = torch.cat([pos2, neg2])
        loss2 = nn.BCEWithLogitsLoss()(pred2, lbl2)
        loss2.backward()
        opt2.step()
    auc_gcn = auc_on_graph(idx, x, gcn, max_samples=100000)
    auc_cann = auc_on_graph(idx, x, cann, K, max_samples=100000)
    return {"GCN": round(auc_gcn, 3), "CANN": round(auc_cann, 3), "nodes": n2, "edges": int(m)}

def main():
    out_path = os.path.abspath(os.path.join("data", "results_link.json"))
    res = {"Cora": run_citation()}
    fb_path = os.path.abspath(os.path.join("data", "real", "social", "facebook", "edges.txt"))
    tw_path = os.path.abspath(os.path.join("data", "real", "social", "twitter", "edges.txt"))
    if os.path.exists(fb_path):
        res["Facebook"] = run_social("Facebook", fb_path)
    if os.path.exists(tw_path):
        res["Twitter"] = run_social("Twitter", tw_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
