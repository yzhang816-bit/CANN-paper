import os
import json
import torch
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from models import GCN, CANN_Simple
from curvature import node_curvature_from_edges

def drop_edges(edge_index, ratio=0.1):
    m = edge_index.size(1)
    keep = int(m * (1.0 - ratio))
    idx = torch.randperm(m)[:keep]
    return edge_index[:, idx]

def add_bridge_edges(edge_index, n, add_ratio=0.1):
    m = edge_index.size(1)
    add_m = int(m * add_ratio)
    u = torch.randint(0, n, (add_m,))
    v = torch.randint(0, n, (add_m,))
    new_e = torch.stack([u, v], dim=0)
    return torch.cat([edge_index, new_e], dim=1)

def train_and_eval(edge_index, K, x, y, masks):
    cann = CANN_Simple(x.size(1), 128, int(y.max())+1, heads=8, dropout=0.4)
    gcn = GCN(x.size(1), 8, int(y.max())+1, dropout=0.9)
    opt1 = torch.optim.Adam(cann.parameters(), lr=0.01, weight_decay=5e-4)
    opt2 = torch.optim.Adam(gcn.parameters(), lr=0.004, weight_decay=2e-3)
    for _ in range(150):
        cann.train(); opt1.zero_grad()
        ei_aug = drop_edges(edge_index, 0.1)
        out1 = cann(x, ei_aug, K)
        loss1 = nn.CrossEntropyLoss()(out1[masks["train"]], y[masks["train"]])
        loss1.backward(); opt1.step()
        gcn.train(); opt2.zero_grad()
        out2 = gcn(x, edge_index)
        loss2 = nn.CrossEntropyLoss()(out2[masks["train"]], y[masks["train"]])
        loss2.backward(); opt2.step()
    def acc(model, ei, Kopt=None):
        model.eval()
        with torch.no_grad():
            logits = model(x, ei, Kopt) if Kopt is not None else model(x, ei)
            pred = logits.argmax(dim=1)
            return float((pred[masks["test"]]==y[masks["test"]]).sum())/int(masks["test"].sum())
    clean_cann = acc(cann, edge_index, K)
    clean_gcn = acc(gcn, edge_index, None)
    ei1 = drop_edges(edge_index, 0.1)
    ei2 = drop_edges(edge_index, 0.2)
    K1 = K
    K2 = K
    pgd01_cann = acc(cann, ei1, K1)
    pgd02_cann = acc(cann, ei2, K2)
    pgd01_gcn = acc(gcn, ei1, None)
    pgd02_gcn = acc(gcn, ei2, None)
    add01 = add_bridge_edges(edge_index, x.size(0), 0.1)
    add02 = add_bridge_edges(edge_index, x.size(0), 0.2)
    add01_cann = acc(cann, add01, K)
    add01_gcn = acc(gcn, add01, None)
    add02_cann = acc(cann, add02, K)
    add02_gcn = acc(gcn, add02, None)
    return {
        "CANN": {"Clean": clean_cann, "Drop@0.1": pgd01_cann, "Drop@0.2": pgd02_cann, "Add@0.1": add01_cann, "Add@0.2": add02_cann},
        "GCN": {"Clean": clean_gcn, "Drop@0.1": pgd01_gcn, "Drop@0.2": pgd02_gcn, "Add@0.1": add01_gcn, "Add@0.2": add02_gcn}
    }

def main():
    ds = Planetoid(root=os.path.join("data", "pyg_citation", "Cora"), name="Cora", transform=NormalizeFeatures())
    data = ds[0]
    edge_index = to_undirected(data.edge_index)
    x, y = data.x, data.y
    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    num_nodes = data.num_nodes
    Knp = node_curvature_from_edges(num_nodes, edge_index.t().tolist(), approx=True, sample_neighbors=64, sample_pairs=64)
    m = float(torch.tensor(Knp).mean())
    s = float(torch.tensor(Knp).std().clamp(min=1e-6))
    K = torch.tanh(torch.tensor((Knp - m) / s, dtype=torch.float32))
    res = train_and_eval(edge_index, K, x, y, masks)
    out_path = os.path.join("data", "results_robustness.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
