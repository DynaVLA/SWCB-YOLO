# SWCB-YOLO

**Morphology-aware supervision improves real-time wind turbine blade defect localization in unstructured natural environments**

This repository contains the official implementation of SWCB-YOLO, a lightweight edge detector
for wind-turbine-blade defect inspection built on the YOLOv11n baseline. SWCB-YOLO targets the
three practical failure modes of general detectors in operating wind farms: high-frequency
background clutter, loss of slender high-aspect-ratio crack features, and curved or bifurcated
image-space defect geometries.

The method introduces three components:

1. **CA-Shape-IoU** — a *morphology-derived, training-only* localization loss. Bézier-curvature
   weights, a Voronoi skeleton-distance field, and an elongation constraint are precomputed
   offline from polygon masks and used to regularize the standard four-parameter box during
   training only, so they correct bounding-box drift at **no inference cost**.
2. **AS-Swin2** — an Asymmetric Strip Swin-TransformerV2 backbone block whose orthogonal strip
   windows give anisotropic receptive fields aligned with the crack propagation axis.
3. **FS-DDA** — a Frequency-Spatial Dual-Domain Attention module that pools each channel's DCT
   magnitude into a spectral descriptor, predicts one scalar weight per channel, and then applies
   large-receptive-field spatial attention. It is frequency-informed channel recalibration, not
   independent filtering of frequency bands.

> On the paper's field dataset, SWCB-YOLO reaches **89.0% mAP@50** and **52.8% mAP@50:95** with
> **3.70 M parameters at 35 FPS** on a Jetson Xavier NX.

---

## Repository layout

```
.
├── ultralytics/                       # YOLOv11 codebase (Ultralytics 8.3.0 base) with SWCB-YOLO additions
│   ├── nn/modules/swcb.py             # AS-Swin2 and FS-DDA modules (this work)
│   ├── utils/ca_shape_iou.py          # CA-Shape-IoU morphological loss terms (this work)
│   ├── utils/ca_fields_provider.py    # runtime loader for the offline morphology caches (this work)
│   ├── utils/metrics.py               # bbox_iou: + CAShapeIoU base term (modified)
│   ├── utils/loss.py                  # BboxLoss / v8DetectionLoss: + CA-Shape-IoU wiring (modified)
│   ├── nn/tasks.py                    # parser registration for ASSwin2 / FS_DDA (modified)
│   ├── cfg/default.yaml               # + cashapeiou / ca_fields hyperparameters (modified)
│   └── data/ca_shape_fields.py        # offline skeleton / curvature / Voronoi field generator (this work)
├── yaml/
│   ├── swcb_yolo.yaml                 # full SWCB-YOLO architecture (Figure 2)
│   └── ablation/                      # per-component ablation configs (Table 4 rows)
├── configs/datasets/                  # dataset configs for the open cross-domain benchmarks
├── tools/prepare_ca_fields.py         # one-shot offline field precomputation for a dataset
├── train.py / val.py / predict.py     # training, evaluation and inference entry points
├── data.yaml                          # dataset template (crack / spalling / dirt)
├── requirements.txt
└── setup.sh                           # environment setup + base overlay
```

## Installation

SWCB-YOLO is built on **Ultralytics 8.3.0**. To keep this repository focused on the
contribution, the unmodified parts of the Ultralytics data pipeline are taken from the pinned
upstream release and the SWCB-YOLO files are overlaid on top. `setup.sh` automates this:

```bash
git clone https://github.com/DynaVLA/SWCB-YOLO.git
cd SWCB-YOLO
bash setup.sh           # creates .venv, installs deps, overlays the upstream data pipeline
```

Manual installation (equivalent):

```bash
pip install ultralytics==8.3.0          # known-good base (provides the data pipeline)
pip install -r requirements.txt         # SWCB-YOLO runtime dependencies
# copy the upstream data package into this repo so the local ultralytics/ is complete:
python - <<'PY'
import os, shutil, ultralytics
src = os.path.join(os.path.dirname(ultralytics.__file__), "data")
shutil.copytree(src, "ultralytics/data", dirs_exist_ok=True)
PY
```

Key dependencies: PyTorch >= 1.12, `timm` (SwinV2 backbone), `scikit-image` and `scipy`
(offline medial-axis skeletonization and curvature fitting), OpenCV.

## Dataset format

Standard Ultralytics detection layout:

```
datasets/blade/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt
```

Edit `data.yaml` to point at your dataset. The wind-turbine-blade dataset in the paper is
operationally restricted (see the paper's Data availability statement); the configs ship with
the three damage categories used there (`crack`, `spalling`, `dirt`).

**For CA-Shape-IoU**, the labels must be *polygon* (segmentation) labels — one object per line as
`<class> x1 y1 x2 y2 ... xk yk` with coordinates normalized to `[0, 1]` — because the curvature
and skeleton fields are derived from the polygon masks.

## Quick start

### 1. (Optional but recommended) Precompute the CA-Shape-IoU morphology fields

```bash
python tools/prepare_ca_fields.py \
    --labels-root datasets/blade/labels \
    --out-root    datasets/blade/ca_fields \
    --imgsz 1280 --splits train val
```

This writes `datasets/blade/ca_fields/<split>/<image_stem>.npz` caches containing the
medial-axis skeleton, the Bézier-fitted radius of curvature, and the elongation descriptor for
each annotated object. If you skip this step, CA-Shape-IoU gracefully falls back to its CIoU
base term and training still runs (without the morphological gains).

### 2. Train

Full SWCB-YOLO with the CA-Shape-IoU loss:

```bash
python train.py \
    --model yaml/swcb_yolo.yaml \
    --data  data.yaml \
    --imgsz 1280 --epochs 100 --batch 16 --optimizer AdamW \
    --cashapeiou --ca-fields datasets/blade/ca_fields/train
```

The optimizer defaults follow the paper's Implementation Details (AdamW, `lr0=1e-3`,
`weight_decay=0.05`, cosine schedule, 100 epochs); SGD tends to oscillate in the asymmetric
attention layers.

### 3. Evaluate

```bash
python val.py --weights runs/detect/swcb_yolo/weights/best.pt --data data.yaml --imgsz 1280
```

Reports mAP@50, mAP@50:95, precision and recall under the COCO-style protocol (confidence 0.25,
NMS IoU 0.7).

### 4. Predict

```bash
python predict.py --weights runs/detect/swcb_yolo/weights/best.pt --source test/ --imgsz 1280 --save
```

## Reproducing the ablation study (Table 4)

Each component can be enabled independently. The configs in `yaml/ablation/` map directly to the
ablation rows; CA-Shape-IoU is toggled by the `--cashapeiou` flag rather than the architecture.

| Configuration | Command |
| --- | --- |
| YOLOv11n baseline | `python train.py --model yaml/ablation/yolov11n_baseline.yaml --data data.yaml` |
| + AS-Swin2 | `python train.py --model yaml/ablation/yolov11n_asswin2.yaml --data data.yaml` |
| + FS-DDA | `python train.py --model yaml/ablation/yolov11n_fsdda.yaml --data data.yaml` |
| + CA-Shape-IoU | `python train.py --model yaml/ablation/yolov11n_baseline.yaml --data data.yaml --cashapeiou --ca-fields datasets/blade/ca_fields/train` |
| **Full SWCB-YOLO** | `python train.py --model yaml/swcb_yolo.yaml --data data.yaml --cashapeiou --ca-fields datasets/blade/ca_fields/train` |

For the identical-protocol comparison, train every configuration from scratch with the same
splits, input size, optimizer, schedule, batch size and epoch budget.

## Cross-domain benchmarks

Dataset configs for all five open benchmarks used in the paper's cross-domain evaluation are in
`configs/datasets/`:

| Benchmark | Configuration | Fixed split and evaluation subset | Detection labels used in the paper |
| --- | --- | --- | --- |
| DTU/NordTank blade inspection | `dtu_blade.yaml` | released 70:15:15 train/validation/test partition; report test | visible surface-damage boxes merged into one `damage` class |
| WTBD multiclass UAV blade defects | `wtbd.yaml` | provider's `train_val_test_split.txt` (70:15:15); report test | six released PASCAL VOC classes converted to YOLO boxes |
| NEU-DET | `neu_det.yaml` | released 1,440/360 train/test partition; reserve a class-stratified 10% of train for validation with seed 42; report test | six steel-surface defect classes |
| DAGM 2007 | `dagm.yaml` | released equal-sized Train/Test partitions within each texture class; reserve 10% of Train for validation with seed 42; report Test | defect regions evaluated as one detection class |
| VisDrone-2019 | `visdrone.yaml` | official 6,471/548/1,610 train/validation/test-dev partition; report test-dev | ten aerial object classes |

The source benchmarks differ in their original annotation formats and granularity. For the common
detection protocol used in the paper, all annotations were standardized as horizontal bounding
boxes. No pseudo-polygons or morphology fields were generated from boxes or weak spatial labels;
therefore, the morphology-dependent terms of CA-Shape-IoU were disabled and the regression
objective reduced to CIoU. The paper uses `640x640` input for every cross-domain run. The precise
partition rules are also recorded in `configs/datasets/SPLIT_PROTOCOL.md`.

```bash
python train.py --model yaml/swcb_yolo.yaml --data configs/datasets/dtu_blade.yaml --imgsz 640
python train.py --model yaml/swcb_yolo.yaml --data configs/datasets/wtbd.yaml --imgsz 640
python train.py --model yaml/swcb_yolo.yaml --data configs/datasets/neu_det.yaml --imgsz 640
python train.py --model yaml/swcb_yolo.yaml --data configs/datasets/dagm.yaml --imgsz 640
python train.py --model yaml/swcb_yolo.yaml --data configs/datasets/visdrone.yaml --imgsz 640
```

## Method details

### CA-Shape-IoU loss

The total loss is `L = L_Base + γ · L_Morph` with `γ = 0.5`, where `L_Base` is the CIoU
alignment term and the normalized morphological component is
`L_Morph = λ1·L̂_Curve + λ2·L̂_Voronoi + λ3·L̂_Ratio`. The default per-term weights are
`(λ1, λ2, λ3) = (1, 1, 1)`.

- **`L̂_Curve`** — curvature-weighted SmoothL1 on sampled box-boundary keypoints, normalized by
  the corresponding ground-truth box diagonal. It uses the
  adaptive weight `w_i = 1 + β · exp(−R_i / τ_c)` (`β = 2`, `τ_c = 5`), so high-curvature regions
  receive up to `1 + β = 3×` the gradient while staying bounded.
- **`L̂_Voronoi`** — truncated squared distance from predicted-box boundary points to the GT crack
  skeleton, `min(d_j², d_max²) / d_max²`, with `d_max = 10 px`.
- **`L̂_Ratio`** — a dimensionless elongation constraint active only for strongly elongated targets
  (GT aspect ratio `> r_th = 8`), implemented with a smooth-`L1` surrogate.

All polygon-derived quantities are computed **offline**; the detection head and inference cost
are unchanged. See `ultralytics/utils/ca_shape_iou.py` and `ultralytics/data/ca_shape_fields.py`.

### AS-Swin2

Augments SwinV2 scaled-cosine windowed attention with two orthogonal strip-window branches
(`1×M²` and `M²×1`) alongside the square `M×M` window. The branches share Q/K/V projections but
use separate directional Log-CPB position-bias generators; their outputs are summed so the
square branch supplies isotropic local context and the strip branches inject axis-aligned
long-range dependencies. See `ultralytics/nn/modules/swcb.py` (`ASSwin2`).

### FS-DDA

Cascades, on a single backbone feature: (1) 2D-DCT spectral channel modulation, (2) a no-bottleneck
full-rank channel attention (CAM), and (3) a large-kernel `7×7` spatial attention (SAM), so
frequency reweighting and spatial geometric refinement act in sequence. See
`ultralytics/nn/modules/swcb.py` (`FS_DDA`).

## Edge deployment

After FP32 training, export to ONNX and build a TensorRT FP16 engine for the Jetson Xavier NX:

```bash
yolo export model=runs/detect/swcb_yolo/weights/best.pt format=onnx imgsz=1280
# then build a TensorRT FP16 engine with trtexec / the TensorRT API on the target device
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{cao2026swcbyolo,
  title   = {Morphology-aware supervision improves real-time wind turbine blade defect localization in unstructured natural environments},
  author  = {Cao, Bingyu and Zhou, Peng and Kan, Mingqi and Chen, Wei and Wang, Yingchao},
  year    = {2026}
}
```

## License

This project is released under the **AGPL-3.0 license**, inherited from the Ultralytics codebase
it builds upon. See [LICENSE](LICENSE).

## Acknowledgements

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLOv11) and the
SwinV2 backbone from [`timm`](https://github.com/huggingface/pytorch-image-models). The
skeleton/curvature pipeline uses [scikit-image](https://scikit-image.org/) and
[SciPy](https://scipy.org/).
