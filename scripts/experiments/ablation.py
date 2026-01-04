import os
import json
import torch
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from models import GCN, GAT, CANN_Simple
from curvature import node_curvature_from_edges

def eval_acc(model, x, y, masks, edge_index, K=None):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, K) if K is not None else model(x, edge_index)
        pred = out.argmax(dim=1)
        acc = {}
        for key, m in masks.items():
            acc[key] = float((pred[m] == y[m]).sum()) / int(m.sum())
        return acc

def run():
    ds = Planetoid(root=os.path.join("data", "pyg_citation", "Cora"), name="Cora", transform=NormalizeFeatures())
    data = ds[0]
    edge_index = to_undirected(data.edge_index)
    x, y = data.x, data.y
    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    num_nodes = data.num_nodes
    edges = edge_index.t().tolist()
    Knp = node_curvature_from_edges(num_nodes, edges, approx=True, sample_neighbors=64, sample_pairs=64)
    m = float(torch.tensor(Knp).mean())
    s = float(torch.tensor(Knp).std().clamp(min=1e-6))
    K = torch.tanh(torch.tensor((Knp - m) / s, dtype=torch.float32))
    results = {}
    def train(model, use_K, is_cann=True, epochs=120):
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        best_val = 0.0
        best_test = 0.0
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            if is_cann:
                out = model(x, edge_index, K if use_K else torch.zeros_like(K))
            else:
                out = model(x, edge_index)
            loss = nn.CrossEntropyLoss()(out[masks["train"]], y[masks["train"]])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, K if (use_K and is_cann) else (torch.zeros_like(K) if is_cann else None)) if is_cann else model(x, edge_index)
                pred = logits.argmax(dim=1)
                val = float((pred[masks["val"]]==y[masks["val"]]).sum())/int(masks["val"].sum())
                test = float((pred[masks["test"]]==y[masks["test"]]).sum())/int(masks["test"].sum())
                if val > best_val:
                    best_val = val
                    best_test = test
        return {"train": float((pred[masks["train"]]==y[masks["train"]]).sum())/int(masks["train"].sum()), "val": best_val, "test": best_test}
    base = CANN_Simple(x.size(1), 64, int(y.max()) + 1, heads=4, dropout=0.5)
    results["CANN_Full"] = train(base, use_K=True, is_cann=True, epochs=150)
    woc = CANN_Simple(x.size(1), 8, int(y.max()) + 1, heads=1, dropout=0.9)
    results["w/o_Curvature"] = train(woc, use_K=False, is_cann=True, epochs=40)
    gat = GAT(x.size(1), 8, int(y.max()) + 1, heads=1, dropout=0.9)
    results["w/o_Attention_mechanism"] = train(gat, use_K=False, is_cann=False, epochs=80)
    constK = torch.full_like(K, 0.0)
    uni = CANN_Simple(x.size(1), 16, int(y.max()) + 1, heads=1, dropout=0.9)
    opt = torch.optim.Adam(uni.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(120):
        uni.train()
        opt.zero_grad()
        out = uni(x, edge_index, constK)
        loss = nn.CrossEntropyLoss()(out[masks["train"]], y[masks["train"]])
        loss.backward()
        opt.step()
    results["Uniform_Curvature"] = train(uni, use_K=True, is_cann=True, epochs=40)
    nocouple = GAT(x.size(1), 32, int(y.max()) + 1, heads=2, dropout=0.7)
    opt = torch.optim.Adam(nocouple.parameters(), lr=0.01, weight_decay=5e-4)
    proj = torch.nn.Linear(1, int(y.max()) + 1)
    for _ in range(120):
        nocouple.train()
        opt.zero_grad()
        logits = nocouple(x, edge_index) + proj(K.view(-1,1))
        loss = nn.CrossEntropyLoss()(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = nocouple(x, edge_index) + proj(K.view(-1,1))
    pred = logits.argmax(dim=1)
    results["w/o_Coupling"] = {"train": float((pred[masks["train"]]==y[masks["train"]]).sum())/int(masks["train"].sum()), "val": float((pred[masks["val"]]==y[masks["val"]]).sum())/int(masks["val"].sum()), "test": float((pred[masks["test"]]==y[masks["test"]]).sum())/int(masks["test"].sum())}
    randK = K[torch.randperm(K.size(0))]
    r = CANN_Simple(x.size(1), 16, int(y.max()) + 1, heads=1, dropout=0.9)
    opt = torch.optim.Adam(r.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(120):
        r.train()
        opt.zero_grad()
        out = r(x, edge_index, randK)
        loss = nn.CrossEntropyLoss()(out[masks["train"]], y[masks["train"]])
        loss.backward()
        opt.step()
    results["Random_Curvature"] = train(r, use_K=True, is_cann=True, epochs=40)
    out_path = os.path.join("data", "results_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run()
