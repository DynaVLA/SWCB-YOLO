# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train SWCB-YOLO on a wind-turbine-blade (or any) detection dataset.

Examples
--------
Train the full SWCB-YOLO model with the CA-Shape-IoU loss and precomputed morphology fields::

    python train.py \
        --model yaml/swcb_yolo.yaml \
        --data  data.yaml \
        --imgsz 1280 --epochs 100 --batch 16 --optimizer AdamW \
        --cashapeiou --ca-fields /path/to/ca_fields/train

Train an ablation variant (e.g. AS-Swin2 backbone only)::

    python train.py --model yaml/ablation/yolov11n_asswin2.yaml --data data.yaml

Reproduce the plain YOLOv11n baseline::

    python train.py --model yaml/ablation/yolov11n_baseline.yaml --data data.yaml

Notes
-----
* The optimizer defaults to AdamW with lr0=1e-3 and weight_decay=0.05, matching the paper's
  Implementation Details; SGD tends to oscillate in the asymmetric attention layers.
* ``--cashapeiou`` enables the Curvature-Aware Shape IoU loss. Its morphological terms also
  require ``--ca-fields`` to point at the offline cache produced by ``tools/prepare_ca_fields.py``.
  Without the cache the loss falls back to its CIoU base term and training still proceeds.
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train SWCB-YOLO.")
    parser.add_argument("--model", type=str, default="yaml/swcb_yolo.yaml",
                        help="model architecture YAML or a .pt checkpoint")
    parser.add_argument("--data", type=str, default="data.yaml", help="dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1280, help="training image size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--device", type=str, default="0", help="cuda device(s) or 'cpu'")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="optimizer weight decay")
    parser.add_argument("--name", type=str, default="swcb_yolo", help="run name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # CA-Shape-IoU
    parser.add_argument("--cashapeiou", action="store_true",
                        help="enable the Curvature-Aware Shape IoU loss")
    parser.add_argument("--ca-fields", type=str, default=None,
                        help="directory of offline CA-Shape-IoU morphology .npz caches")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        name=args.name,
        seed=args.seed,
        cashapeiou=args.cashapeiou,
        ca_fields=args.ca_fields,
    )


if __name__ == "__main__":
    main()
