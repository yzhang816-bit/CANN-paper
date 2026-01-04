import os
import json
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from curvature import node_curvature_from_edges

def main():
    ds = Planetoid(root=os.path.join("data", "pyg_citation", "Cora"), name="Cora", transform=NormalizeFeatures())
    data = ds[0]
    edge_index = to_undirected(data.edge_index)
    num_nodes = data.num_nodes
    edges = edge_index.t().tolist()
    K = node_curvature_from_edges(num_nodes, edges, approx=True, sample_neighbors=64, sample_pairs=64)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    Kt = torch.tensor(K, dtype=torch.float32)
    meanK = float(Kt.mean()); stdK = float(Kt.std())
    pos = int((Kt > 0.2).sum()); neg = int((Kt < -0.2).sum()); zero = num_nodes - pos - neg
    corr = float(torch.corrcoef(torch.stack([Kt, deg]))[0,1])
    bins = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
    hist = []
    for i in range(len(bins)-1):
        l, r = bins[i], bins[i+1]
        cnt = int(((Kt >= l) & (Kt < r)).sum())
        hist.append({"bin": f"[{l},{r})", "count": cnt})
    out = {
        "mean_curvature": round(meanK, 3),
        "std_curvature": round(stdK, 3),
        "positive_nodes": pos,
        "negative_nodes": neg,
        "near_zero_nodes": zero,
        "corr_degree_curvature": round(corr, 3),
        "histogram": hist
    }
    with open(os.path.join("data", "results_interpretability.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
