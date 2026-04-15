from pathlib import Path

import numpy as np


def find_smallest_npz(root: Path) -> Path:
    npz_files = [p for p in root.rglob("*.npz") if p.is_file()]
    if not npz_files:
        raise FileNotFoundError("No .npz files found under data/")
    return min(npz_files, key=lambda p: (p.stat().st_size, str(p)))


def main() -> None:
    data_root = Path("data")
    smallest = find_smallest_npz(data_root)

    print(f"Using file: {smallest}")
    print(f"File size: {smallest.stat().st_size} bytes")

    with np.load(smallest, allow_pickle=False) as d:
        missing = [k for k in ("x", "y", "mask") if k not in d.files]
        if missing:
            raise KeyError(f"Missing expected arrays: {missing}; present keys={d.files}")

        x = d["x"]
        y = d["y"]
        mask = d["mask"]

    if not (len(x) == len(y) == len(mask)):
        raise ValueError("x, y, and mask lengths do not match")

    total = len(x)
    n = min(5, total)
    rng = np.random.default_rng(123)
    idxs = rng.choice(total, size=n, replace=False)

    print(f"x shape={x.shape}, dtype={x.dtype}")
    print(f"y shape={y.shape}, dtype={y.dtype}")
    print(f"mask shape={mask.shape}, dtype={mask.dtype}")
    print(f"Printing {n} random sample indices: {idxs.tolist()}")

    for i, idx in enumerate(idxs, start=1):
        print(f"\n--- Sample {i}/{n} (index={int(idx)}) ---")
        print("x:")
        print(x[idx])
        print("y:")
        print(y[idx])
        print("mask:")
        print(mask[idx])


if __name__ == "__main__":
    main()