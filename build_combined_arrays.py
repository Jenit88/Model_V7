#!/usr/bin/env python3
"""Concatenate the original V7 arrays with the 3000_new_images arrays.

    train = old train (4,680) + new train (11,160)
    val   = old val   (1,048) + new val   (2,080)
    test  = new test  (664)        # the old 432-image test stays in OLD

Every split keeps its images where they were: train stays train, val stays
val. The old arrays are only read. A per-split index map is written to
combined_sources.json so any combined index traces back to its source file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

OLD = Path("/home/u117134c/data/pcb_v62_arrays_repaired")
NEW = Path("/home/u117134c/data/pcb_v7_new3000_arrays")
OUT = Path("/home/u117134c/data/pcb_v7_combined_arrays")
PLAN = {"train": (OLD, NEW), "val": (OLD, NEW), "test": (NEW,)}
KINDS = ("X", "Y_semantic", "Y_instance")
CHUNK = 256


def records(root: Path, split: str) -> list[dict]:
    path = root / f"{split}_records.json"
    return json.loads(path.read_text()) if path.is_file() else []


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[dict]] = {}
    for split, roots in PLAN.items():
        sources[split] = []
        for kind in KINDS:
            parts = [np.load(r / f"{kind}_{split}.npy", mmap_mode="r") for r in roots]
            dtypes = {p.dtype for p in parts}
            shapes = {p.shape[1:] for p in parts}
            if len(dtypes) != 1 or len(shapes) != 1:
                sys.exit(f"{kind}_{split}: sources disagree {dtypes} {shapes}")
            total = sum(len(p) for p in parts)
            out = open_memmap(OUT / f"{kind}_{split}.npy", mode="w+",
                              dtype=parts[0].dtype, shape=(total, *parts[0].shape[1:]))
            offset = 0
            for root, part in zip(roots, parts):
                for start in range(0, len(part), CHUNK):
                    block = part[start:start + CHUNK]
                    out[offset + start:offset + start + len(block)] = block
                if kind == "X":
                    recs = records(root, split)
                    sources[split].append({
                        "source": str(root), "combined_start": offset,
                        "count": len(part),
                        "files": [r.get("image") for r in recs] if recs else None,
                    })
                offset += len(part)
            out.flush()
            del out
            print(f"{kind}_{split}: {total:,} rows", flush=True)

    (OUT / "combined_sources.json").write_text(json.dumps(sources, indent=1))
    meta_old = json.loads((OLD / "instance_dataset_metadata.json").read_text())
    meta_new = json.loads((NEW / "instance_dataset_metadata.json").read_text())
    meta = {k: v for k, v in meta_new.items() if k != "splits"}
    meta["array_dir"] = str(OUT)
    meta["dataset_root"] = "combined: see combined_sources.json"
    meta["splits"] = []
    for split, roots in PLAN.items():
        rows = [next(s for s in (meta_old if r == OLD else meta_new)["splits"]
                     if s["split"] == split) for r in roots]
        meta["splits"].append({
            "split": split,
            "images": sum(r["images"] for r in rows),
            "instances": sum(r["instances"] for r in rows),
            "parts": [{"source": str(r), "images": row["images"],
                       "instances": row["instances"]}
                      for r, row in zip(roots, rows)],
        })
    (OUT / "instance_dataset_metadata.json").write_text(json.dumps(meta, indent=2))
    print("combined arrays written to", OUT)


if __name__ == "__main__":
    main()
