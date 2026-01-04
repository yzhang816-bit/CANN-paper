import os
import json
import subprocess
import sys

def run(cmd, cwd):
    p = subprocess.Popen(cmd, cwd=cwd, shell=True)
    p.wait()
    return p.returncode

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exp_dir = os.path.join(root, "experiments")
    datasets_dir = os.path.join(root, "datasets")
    subprocess.Popen(f"{sys.executable} prepare_datasets.py --real", cwd=datasets_dir, shell=True).wait()
    run(f"{sys.executable} train_node.py", exp_dir)
    run(f"{sys.executable} train_link.py", exp_dir)
    run(f"{sys.executable} train_graph.py", exp_dir)
    out = {}
    base = os.path.abspath(os.path.join(root, "..", "data"))
    for name in ["results_node.json", "results_link.json", "results_graph.json"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                out[name] = json.load(f)
    with open(os.path.join(base, "results_all.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
