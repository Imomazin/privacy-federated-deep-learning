"""
Dataset acquisition and integrity verification for the RAPSA-FL study.

Downloads the three datasets from sources reachable without institutional
access, then verifies each against published checksums or published summary
statistics. Exits non-zero if any check fails.

Usage:  python3 acquire-datasets.py --out ../data/raw
"""

import argparse
import hashlib
import os
import sys
import urllib.request

MNIST_BASE = "https://raw.githubusercontent.com/fgnt/mnist/master"
FASHION_BASE = ("https://raw.githubusercontent.com/zalandoresearch/"
                "fashion-mnist/master/data/fashion")
CREDIT_URL = ("https://raw.githubusercontent.com/Knchna/Credit_Card_Default_Prediction/"
              "master/data/default%20of%20credit%20card%20clients.xls")

# Official MNIST checksums as published with the original distribution.
MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

# Official Fashion-MNIST checksums from the Zalando Research repository.
FASHION_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

# Published summary statistics for the Taiwan credit default data, Yeh and Lien 2009.
CREDIT_EXPECTED = {"rows": 30000, "cols": 25, "defaults": 6636}


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def idx_files(base, out_dir, expected, label):
    ok = True
    for name, want in expected.items():
        path = fetch(f"{base}/{name}", os.path.join(out_dir, name))
        got = md5(path)
        status = "pass" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {label:14s} {name:30s} {os.path.getsize(path):>9d} B  md5 {got}  {status}")
    return ok


def credit_file(out_dir):
    path = fetch(CREDIT_URL, os.path.join(out_dir, "credit.xls"))
    import pandas as pd

    d = pd.read_excel(path, header=1)
    defaults = int(d["default payment next month"].sum())
    checks = {
        "rows": d.shape[0] == CREDIT_EXPECTED["rows"],
        "cols": d.shape[1] == CREDIT_EXPECTED["cols"],
        "defaults": defaults == CREDIT_EXPECTED["defaults"],
        "ids unique": d["ID"].nunique() == CREDIT_EXPECTED["rows"],
    }
    print(f"  {'credit':14s} {'credit.xls':30s} {os.path.getsize(path):>9d} B  "
          f"md5 {md5(path)}")
    print(f"                 rows={d.shape[0]} cols={d.shape[1]} defaults={defaults} "
          f"rate={defaults / d.shape[0]:.4f}")
    for k, v in checks.items():
        print(f"                 {k:12s} {'pass' if v else 'FAIL'}")
    return all(checks.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../data/raw")
    a = p.parse_args()
    out = os.path.abspath(a.out)
    print(f"Acquiring datasets into {out}\n")
    results = {
        "mnist": idx_files(MNIST_BASE, os.path.join(out, "mnist"), MNIST_MD5, "mnist"),
        "fashion": idx_files(FASHION_BASE, os.path.join(out, "fashion"), FASHION_MD5, "fashion"),
        "credit": credit_file(out),
    }
    print()
    for k, v in results.items():
        print(f"{k:10s} {'verified' if v else 'VERIFICATION FAILED'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
