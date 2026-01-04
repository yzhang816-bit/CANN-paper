import os
import math
import argparse
import random
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_edges(edges, path):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in edges:
            if u == v:
                continue
            f.write(f"{u}\t{v}\n")

def generate_sbm(n=1000, k=5, p_in=0.3, p_out=0.01, seed=42):
    rng = np.random.default_rng(seed)
    sizes = [n // k] * k
    for i in range(n % k):
        sizes[i] += 1
    cum = np.cumsum([0] + sizes)
    comm = np.zeros(n, dtype=np.int32)
    for i in range(k):
        comm[cum[i]:cum[i+1]] = i
    edges = []
    for u in range(n):
        cu = comm[u]
        for v in range(u + 1, n):
            cv = comm[v]
            p = p_in if cu == cv else p_out
            if rng.random() < p:
                edges.append((u, v))
                edges.append((v, u))
    return edges

def generate_ba(n=1000, avg_degree=10, seed=42):
    rng = random.Random(seed)
    m = max(1, avg_degree // 2)
    edges = []
    targets = list(range(m))
    repeated_nodes = []
    for i in range(m):
        for j in range(i + 1, m):
            edges.append((i, j))
            edges.append((j, i))
    repeated_nodes.extend([i for i in range(m) for _ in range(m)])
    for new_node in range(m, n):
        new_targets = set()
        while len(new_targets) < m:
            new_targets.add(rng.choice(repeated_nodes))
        for t in new_targets:
            edges.append((new_node, t))
            edges.append((t, new_node))
        repeated_nodes.extend(list(new_targets))
        repeated_nodes.extend([new_node] * (m + 1))
    return edges

def generate_mixed(n_total=1000, k=5, p_in=0.3, p_out=0.01, tree_nodes=200, seed=42):
    rng = np.random.default_rng(seed)
    n_sbm = n_total - tree_nodes
    sbm_edges = generate_sbm(n=n_sbm, k=k, p_in=p_in, p_out=p_out, seed=seed)
    edges = list(sbm_edges)
    roots = rng.choice(n_sbm, size=max(1, tree_nodes // 10), replace=False)
    remaining = tree_nodes
    node_offset = n_sbm
    for r in roots:
        size = min(remaining, rng.integers(8, 20))
        remaining -= size
        if size <= 0:
            break
        parent = r
        for i in range(size):
            u = node_offset + i
            edges.append((parent, u))
            edges.append((u, parent))
            parent = u
        node_offset += size
        if remaining <= 0:
            break
    return edges, node_offset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=os.path.join("data", "synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base = os.path.abspath(args.out)
    ensure_dir(base)
    sbm_dir = os.path.join(base, "sbm")
    hyp_dir = os.path.join(base, "hyperbolic_ba")
    mix_dir = os.path.join(base, "mixed_curvature")
    ensure_dir(sbm_dir)
    ensure_dir(hyp_dir)
    ensure_dir(mix_dir)
    sbm_edges = generate_sbm(n=1000, k=5, p_in=0.3, p_out=0.01, seed=args.seed)
    save_edges(sbm_edges, os.path.join(sbm_dir, "edges.txt"))
    ba_edges = generate_ba(n=1000, avg_degree=10, seed=args.seed)
    save_edges(ba_edges, os.path.join(hyp_dir, "edges.txt"))
    mix_edges, n_total = generate_mixed(n_total=1000, k=5, p_in=0.3, p_out=0.01, tree_nodes=200, seed=args.seed)
    save_edges(mix_edges, os.path.join(mix_dir, "edges.txt"))
    with open(os.path.join(sbm_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write("n=1000 k=5 p_in=0.3 p_out=0.01\n")
    with open(os.path.join(hyp_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write("n=1000 approx=barabasi_albert avg_degree=10\n")
    with open(os.path.join(mix_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"n_total={n_total} sbm_base=800 tree_nodes≈200\n")

if __name__ == "__main__":
    main()
