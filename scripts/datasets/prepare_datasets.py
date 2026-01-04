import os
import argparse
import subprocess
import sys

def run(cmd, cwd):
    p = subprocess.Popen(cmd, cwd=cwd, shell=True)
    p.wait()
    return p.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    scripts_dir = os.path.join(root, "scripts", "datasets")
    if args.all or args.synthetic:
        run(f"{sys.executable} generate_synthetic.py --out {os.path.join(root, 'data', 'synthetic')}", scripts_dir)
    if args.all or args.real:
        run(f"{sys.executable} download_real.py --all --out {os.path.join(root, 'data', 'real')}", scripts_dir)

if __name__ == "__main__":
    main()
