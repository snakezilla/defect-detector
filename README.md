# defect-detector

Visual defect detection for manufacturing. The model learns only from good parts, because a real line has few labelled defects.

The method is PatchCore (Roth et al., CVPR 2022). I wrote it from the paper in about 150 lines of PyTorch (`src/patchcore.py`). It does not use a library.

## How it works
1. A frozen ImageNet network (Wide ResNet-50) gives a feature vector for each small patch of the image. It uses layers 2 and 3, and each patch also sees its 3 x 3 neighbours.
2. A memory bank keeps 10% of the good-part patch features. Greedy k-center selection picks the patches that cover the feature space.
3. At test time, each patch gets a score: the distance to its nearest good patch. The image score is the largest patch score. The patch scores, smoothed, make the heatmap.

## Results on MVTec AD (all 15 categories)

| | Image AUROC | Pixel AUROC |
|---|---|---|
| This repo | 0.988 | 0.979 |
| PatchCore paper (10% coreset) | 0.990 | 0.981 |

Latency is 9 to 24 ms for one 224 px image on an Apple M5 Pro (MPS). Per-category numbers and heatmaps are in `results/mvtec_20261117_225928/`.

| Category | Image | Pixel | Category | Image | Pixel |
|---|---|---|---|---|---|
| bottle | 1.000 | 0.984 | pill | 0.949 | 0.977 |
| cable | 0.997 | 0.985 | screw | 0.966 | 0.988 |
| capsule | 0.973 | 0.987 | tile | 0.985 | 0.953 |
| carpet | 0.990 | 0.988 | toothbrush | 0.997 | 0.987 |
| grid | 0.977 | 0.976 | transistor | 1.000 | 0.976 |
| hazelnut | 1.000 | 0.984 | wood | 0.994 | 0.938 |
| leather | 1.000 | 0.991 | zipper | 0.995 | 0.982 |
| metal_nut | 0.997 | 0.986 | | | |

The weakest categories are pill (0.949) and screw (0.966). Pill defects are small colour specks. Screws sit at a different angle in each image, so a "good" patch can look like a defect.

## Run it
```
python3 -m venv .venv && .venv/bin/pip install torch torchvision scikit-learn pillow numpy tqdm matplotlib
# put MVTec AD in data/mvtec/ (https://www.mvtec.com/company/research/datasets/mvtec-ad, CC BY-NC-SA 4.0)
.venv/bin/python src/run_mvtec.py                        # all categories
.venv/bin/python src/run_mvtec.py --categories screw     # one category
.venv/bin/python -m unittest discover -s tests
```

## Next: real parts on a camera
Phase 2 runs the same model live on 3D-printed parts with a fixed camera on an edge box. It will log precision, recall, latency and false alarms per 100 parts.

## Credits
- PatchCore: K. Roth et al., "Towards Total Recall in Industrial Anomaly Detection," CVPR 2022.
- MVTec AD: P. Bergmann et al., CVPR 2019. The dataset is CC BY-NC-SA 4.0 and is not in this repo. The heatmap images in `results/` are made from it and are shared under the same license.
