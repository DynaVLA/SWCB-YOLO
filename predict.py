# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run SWCB-YOLO inference on images, a directory, or a video.

Example
-------
    python predict.py --weights runs/detect/swcb_yolo/weights/best.pt --source test/ --imgsz 1280
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWCB-YOLO inference.")
    parser.add_argument("--weights", type=str, required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--source", type=str, required=True, help="image / directory / video path")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference image size")
    parser.add_argument("--device", type=str, default="0", help="cuda device(s) or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--save", action="store_true", help="save annotated outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
    )


if __name__ == "__main__":
    main()
