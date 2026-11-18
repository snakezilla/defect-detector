"""Run PatchCore on MVTec AD categories and write results.

Usage: .venv/bin/python src/run_mvtec.py [--categories bottle screw ...] [--coreset 0.1]
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import CATEGORIES, denormalize, load_batch, test_samples, train_images
from patchcore import PatchCore, pick_device

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mvtec"
RESULTS = ROOT / "results"
BATCH = 16


def batches(paths, size):
    for i in range(0, len(paths), size):
        yield load_batch(paths[i:i + size])


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def save_examples(samples, scores, maps, out_path, n=6):
    """Save the highest-scoring defects next to their true masks."""
    order = np.argsort(-scores)
    picks = [i for i in order if samples[i].label == 1][: n - 2] + [i for i in order[::-1] if samples[i].label == 0][:2]
    fig, axes = plt.subplots(3, len(picks), figsize=(2.2 * len(picks), 6.6))
    vmax = float(maps.max())
    for col, i in enumerate(picks):
        s = samples[i]
        axes[0, col].imshow(denormalize(s.image))
        axes[0, col].set_title(f"{s.defect_type}\nscore {scores[i]:.2f}", fontsize=8)
        axes[1, col].imshow(s.mask[0], cmap="gray")
        axes[2, col].imshow(denormalize(s.image))
        axes[2, col].imshow(maps[i, 0], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].text(-20, 112, "image", rotation=90, va="center", fontsize=8)
    axes[1, 0].text(-20, 112, "true defect", rotation=90, va="center", fontsize=8)
    axes[2, 0].text(-20, 112, "model heatmap", rotation=90, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_category(category, device, coreset, out_dir):
    t0 = time.perf_counter()
    model = PatchCore(device, coreset_ratio=coreset)
    train = train_images(DATA, category)
    bank_size = model.fit(batches(train, BATCH))
    sync(device)
    fit_s = time.perf_counter() - t0

    samples = test_samples(DATA, category)
    scores, maps = [], []
    t1 = time.perf_counter()
    for i in range(0, len(samples), BATCH):
        x = torch.stack([s.image for s in samples[i:i + BATCH]])
        sc, mp = model.predict(x)
        scores.append(sc)
        maps.append(mp)
    sync(device)
    infer_s = time.perf_counter() - t1

    scores = torch.cat(scores).numpy()
    maps = torch.cat(maps).numpy()
    labels = np.array([s.label for s in samples])
    masks = torch.stack([s.mask for s in samples]).numpy()

    image_auroc = roc_auc_score(labels, scores)
    pixel_auroc = roc_auc_score(masks.reshape(-1).astype(int), maps.reshape(-1))
    save_examples(samples, scores, maps, out_dir / f"{category}.png")

    # Single-image latency, the number that matters on a line.
    x1 = samples[0].image.unsqueeze(0)
    model.predict(x1)
    sync(device)
    t2 = time.perf_counter()
    for _ in range(10):
        model.predict(x1)
    sync(device)
    latency_ms = (time.perf_counter() - t2) / 10 * 1000

    return {
        "category": category,
        "train_images": len(train),
        "test_images": len(samples),
        "bank_size": bank_size,
        "image_auroc": round(float(image_auroc), 4),
        "pixel_auroc": round(float(pixel_auroc), 4),
        "fit_seconds": round(fit_s, 1),
        "test_ms_per_image_batched": round(infer_s / len(samples) * 1000, 1),
        "latency_ms_single_image": round(latency_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--coreset", type=float, default=0.1)
    args = ap.parse_args()

    device = pick_device()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / f"mvtec_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"device {device}, coreset {args.coreset}, writing to {out_dir}", flush=True)

    rows = []
    for cat in args.categories:
        row = run_category(cat, device, args.coreset, out_dir)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out_dir / "results.json").write_text(json.dumps({"device": str(device), "coreset": args.coreset, "rows": rows}, indent=2))

    mean_img = np.mean([r["image_auroc"] for r in rows])
    mean_pix = np.mean([r["pixel_auroc"] for r in rows])
    print(f"MEAN image AUROC {mean_img:.4f}  pixel AUROC {mean_pix:.4f}  over {len(rows)} categories", flush=True)


if __name__ == "__main__":
    main()
