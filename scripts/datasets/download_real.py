import os
import argparse
import gzip
import shutil
import sys
from urllib.request import urlopen, Request

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch(url, out_path):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)

def decompress_gz(gz_path, out_path):
    with gzip.open(gz_path, "rb") as gzf, open(out_path, "wb") as out:
        shutil.copyfileobj(gzf, out)

def download_planetoid(name, base_dir):
    target = os.path.join(base_dir, "citation", name)
    ensure_dir(target)
    files = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    for fn in files:
        url = f"https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{name}.{fn}"
        out = os.path.join(target, f"ind.{name}.{fn}")
        fetch(url, out)

def download_snap_facebook(base_dir):
    target = os.path.join(base_dir, "social", "facebook")
    ensure_dir(target)
    gz_path = os.path.join(target, "facebook_combined.txt.gz")
    txt_path = os.path.join(target, "edges.txt")
    fetch("http://snap.stanford.edu/data/facebook_combined.txt.gz", gz_path)
    decompress_gz(gz_path, txt_path)

def download_snap_twitter(base_dir):
    target = os.path.join(base_dir, "social", "twitter")
    ensure_dir(target)
    gz_path = os.path.join(target, "twitter_combined.txt.gz")
    txt_path = os.path.join(target, "edges.txt")
    fetch("http://snap.stanford.edu/data/twitter_combined.txt.gz", gz_path)
    decompress_gz(gz_path, txt_path)

def try_pyg_zinc(base_dir):
    try:
        from torch_geometric.datasets import ZINC
        import torch
    except Exception:
        return False
    target = os.path.join(base_dir, "molecular", "zinc")
    ensure_dir(target)
    ds = ZINC(root=target)
    torch.save({"len": len(ds)}, os.path.join(target, "meta.pt"))
    return True

def try_ogb_molhiv(base_dir):
    try:
        from ogb.graphproppred import GraphPropPredDataset
        import torch
    except Exception:
        return False
    target = os.path.join(base_dir, "molecular", "ogb_molhiv")
    ensure_dir(target)
    ds = GraphPropPredDataset(name="ogbg-molhiv", root=target)
    torch.save({"len": len(ds)}, os.path.join(target, "meta.pt"))
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=os.path.join("data", "real"))
    parser.add_argument("--citation", action="store_true")
    parser.add_argument("--social", action="store_true")
    parser.add_argument("--molecular", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    base = os.path.abspath(args.out)
    ensure_dir(base)
    if args.all or args.citation:
        download_planetoid("cora", base)
        download_planetoid("citeseer", base)
        download_planetoid("pubmed", base)
    if args.all or args.social:
        download_snap_facebook(base)
        download_snap_twitter(base)
    if args.all or args.molecular:
        try_pyg_zinc(base)
        try_ogb_molhiv(base)

if __name__ == "__main__":
    main()
