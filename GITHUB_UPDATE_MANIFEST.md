# GitHub update manifest

Copy the files in this package to the repository root while preserving their relative paths.
Existing files with the same paths should be replaced. The added test file should be retained.

## Replace

- README.md
- CITATION.cff
- train.py
- ultralytics/cfg/default.yaml
- ultralytics/utils/ca_shape_iou.py
- ultralytics/utils/loss.py
- configs/datasets/SPLIT_PROTOCOL.md
- configs/datasets/dtu_blade.yaml
- configs/datasets/wtbd.yaml
- configs/datasets/neu_det.yaml
- configs/datasets/dagm.yaml
- configs/datasets/visdrone.yaml

## Add

- tests/test_ca_shape_iou_normalization.py

## What this update changes

- aligns the paper title, citation metadata, metrics, table numbers, and public-dataset protocols;
- removes command placeholders from the README;
- implements the normalized and explicitly weighted CA-Shape-IoU terms described in the manuscript;
- makes nearest-curve assignment a detached forward-pass correspondence;
- keeps the 10-pixel distance threshold in the morphology-cache canvas coordinate system;
- exposes the morphology balance and per-term weights in the training configuration;
- applies the documented AdamW, cosine schedule, warm-up, and final-15-epoch Mosaic shutdown.

Before publishing, run the repository's existing tests and
python -m pytest tests/test_ca_shape_iou_normalization.py in the project environment.
The actual public-dataset split-index files should also be committed if they are available.

