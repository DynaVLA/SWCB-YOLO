# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Evaluate a trained SWCB-YOLO checkpoint and report mAP@50, mAP@50:95, precision and recall.

Example
-------
    python val.py --weights runs/detect/swcb_yolo/weights/best.pt --data data.yaml --imgsz 1280
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SWCB-YOLO.")
    parser.add_argument("--weights", type=str, required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--data", type=str, default="data.yaml", help="dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1280, help="evaluation image size")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="cuda device(s) or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )
    print(f"mAP@50      : {metrics.box.map50:.4f}")
    print(f"mAP@50:95   : {metrics.box.map:.4f}")
    print(f"precision   : {metrics.box.mp:.4f}")
    print(f"recall      : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
