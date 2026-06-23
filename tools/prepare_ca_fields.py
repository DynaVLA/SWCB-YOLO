# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Precompute the offline morphology fields required by the CA-Shape-IoU loss.

This is a thin wrapper around ``ultralytics.data.ca_shape_fields`` that processes the train,
val and test polygon-label splits in one call. Run it once before training with
``--cashapeiou``.

The input labels must be YOLO-format *polygon* (segmentation) labels: one object per line as
``<class> x1 y1 x2 y2 ... xk yk`` with coordinates normalized to [0, 1].

Example
-------
    python tools/prepare_ca_fields.py \
        --labels-root /path/to/dataset/labels \
        --out-root    /path/to/dataset/ca_fields \
        --imgsz 1280 --splits train val

This writes ``<out-root>/<split>/<image_stem>.npz`` caches, which you then pass to training via
``--ca-fields /path/to/dataset/ca_fields/train``.
"""

import argparse
from pathlib import Path

from ultralytics.data.ca_shape_fields import build_dataset_fields


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CA-Shape-IoU morphology fields for a dataset.")
    parser.add_argument("--labels-root", type=str, required=True,
                        help="root containing per-split polygon label folders (e.g. labels/train)")
    parser.add_argument("--out-root", type=str, required=True,
                        help="root where per-split .npz cache folders will be written")
    parser.add_argument("--imgsz", type=int, default=1280, help="rasterization canvas size")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="dataset splits to process")
    return parser.parse_args()


def main():
    args = parse_args()
    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)
    for split in args.splits:
        labels_dir = labels_root / split
        if not labels_dir.is_dir():
            print(f"[prepare_ca_fields] skip '{split}': {labels_dir} not found")
            continue
        out_dir = out_root / split
        print(f"[prepare_ca_fields] processing split '{split}' -> {out_dir}")
        build_dataset_fields(labels_dir, out_dir, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
