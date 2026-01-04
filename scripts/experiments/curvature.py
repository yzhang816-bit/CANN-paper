import numpy as np
import random

def node_curvature_from_edges(num_nodes, edges, approx=False, sample_neighbors=64, sample_pairs=64):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(nei) for nei in adj]
    if not approx:
        sets = [set(nei) for nei in adj]
        k = np.zeros(num_nodes, dtype=np.float32)
        for u in range(num_nodes):
            if deg[u] == 0:
                k[u] = 0.0
                continue
            s = 0.0
            for v in adj[u]:
                du = deg[u]
                dv = deg[v]
                inter = len(sets[u].intersection(sets[v]))
                if du + dv > 0:
                    s += (2.0 * inter) / (du + dv) - 0.5
            k[u] = s / max(1, deg[u])
        return k
    k = np.zeros(num_nodes, dtype=np.float32)
    for u in range(num_nodes):
        du = deg[u]
        if du == 0:
            k[u] = 0.0
            continue
        neis_u = adj[u]
        ns = min(sample_neighbors, du)
        sample_v = random.sample(neis_u, ns) if du > ns else neis_u
        s = 0.0
        for v in sample_v:
            dv = deg[v]
            if dv == 0:
                continue
            neis_v = adj[v]
            sv = min(sample_pairs, du, dv)
            probe = random.sample(neis_u, sv) if du > sv else neis_u
            set_v = set(neis_v) if dv < 10000 else None
            if set_v is None:
                inter = sum(1 for w in probe if w in neis_v)
            else:
                inter = sum(1 for w in probe if w in set_v)
            est_cn = inter * (du / max(1, len(probe)))
            if du + dv > 0:
                s += (2.0 * est_cn) / (du + dv) - 0.5
        k[u] = s / max(1, len(sample_v))
    return k
