import os
import json
import time
import torch
from models import CANN_Simple
from curvature import node_curvature_from_edges
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "datasets"))
from generate_synthetic import generate_sbm, generate_ba

def to_edge_index(edges, n):
    import torch
    idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return idx

def run_sizes():
    sizes = [
        ("Small", 1000, 5000),
        ("Medium", 10000, 50000),
    ]
    res = []
    for name, n, target_m in sizes:
        sbm_edges = generate_sbm(n=n, k=5, p_in=0.3, p_out=0.01, seed=42)
        if len(sbm_edges)//2 > target_m:
            sbm_edges = sbm_edges[:2*target_m]
        m = len(sbm_edges)//2
        t0 = time.time()
        K = node_curvature_from_edges(n, sbm_edges, approx=True, sample_neighbors=16, sample_pairs=16)
        t1 = time.time()
        idx = to_edge_index(sbm_edges, n)
        deg = torch.zeros(n, 1)
        for u, v in sbm_edges:
            deg[u, 0] += 1; deg[v, 0] += 1
        x = (deg / deg.max().clamp(min=1)).float()
        Kt = torch.tanh(torch.tensor((K - float(torch.tensor(K).mean())) / float(torch.tensor(K).std().clamp(min=1e-6)), dtype=torch.float32))
        model = CANN_Simple(1, 32, 32, heads=4, dropout=0.5)
        t2 = time.time()
        for _ in range(1):
            model.train()
            out = model(x, idx, Kt)
        t3 = time.time()
        res.append({"GraphSize": name, "Nodes": n, "Edges": m, "CurvatureTimeSec": round(t1 - t0, 3), "TrainTimePerEpochSec": round(t3 - t2, 3)})
    with open(os.path.join("data", "results_scalability.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_sizes()
