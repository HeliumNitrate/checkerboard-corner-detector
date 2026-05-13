"""
compare_opencv_vs_cnn.py
========================
npz'den N görüntü al.
Her görüntüde OpenCV ve finetuned.pt'yi karşılaştır.
GT yeşil, OpenCV sarı, CNN camgöbeği.

Kullanım:
    python compare_opencv_vs_cnn.py
    python compare_opencv_vs_cnn.py --n 20 --seed 0
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import load_model, predict_heatmap, heatmap_to_corners

REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH  = os.path.join(REPO_DIR, '..', 'OpticalDistortionLab', 'datasets', 'corners.npz')
CKPT_FT   = os.path.join(REPO_DIR, 'checkpoints', 'finetuned.pt')
OUT_DIR   = os.path.join(REPO_DIR, 'test_results', 'opencv_vs_cnn')
IMG_SIZE  = 256

BOARD_SIZES = [
    (7, 11), (11, 7), (7, 3), (3, 7),
    (9, 6),  (6, 9),  (8, 6), (6, 8),
    (7, 6),  (6, 7),  (7, 7),
    (12, 7), (7, 12), (10, 7), (7, 10),
]


def find_corners_opencv(img_gray):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    small = cv2.resize(img_gray, (min(1024, img_gray.shape[1]),
                                  min(1024, img_gray.shape[0])))
    scale = img_gray.shape[1] / small.shape[1]
    for rows, cols in BOARD_SIZES:
        try:
            ok, corners = cv2.findChessboardCornersSB(small, (cols, rows), flags)
        except cv2.error:
            ok = False
        if ok:
            corners = corners.reshape(-1, 2) * scale
            return corners
        ok, corners = cv2.findChessboardCorners(
            small, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(small, corners, (11, 11), (-1, -1), crit)
            corners = corners.reshape(-1, 2) * scale
            return corners
    return np.zeros((0, 2))


def pr_score(pred_rc, gt_rc, tolerance=3.0):
    if len(gt_rc) == 0 or len(pred_rc) == 0:
        return 0.0, 0.0
    matched, tp = set(), 0
    for p in pred_rc:
        dists = np.linalg.norm(gt_rc - p, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= tolerance and j not in matched:
            tp += 1; matched.add(j)
    return tp / len(pred_rc), tp / len(gt_rc)


def draw_comparison(img256, gt_rc, ocv_rc, cnn_rc, title, prec_ocv, rec_ocv, prec_cnn, rec_cnn):
    """İki panel: sol OpenCV, sağ CNN. Her ikisinde GT yeşil nokta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=8)

    for ax, pts, label, color, prec, rec in [
        (axes[0], ocv_rc,  'OpenCV',        'yellow', prec_ocv, rec_ocv),
        (axes[1], cnn_rc,  'finetuned.pt',  'cyan',   prec_cnn, rec_cnn),
    ]:
        ax.imshow(img256, cmap='gray', vmin=0, vmax=255)
        # GT
        if len(gt_rc):
            ax.scatter(gt_rc[:, 1], gt_rc[:, 0],
                       s=30, c='lime', marker='o', linewidths=0,
                       label=f'GT ({len(gt_rc)})', zorder=2)
        # Tahmin
        if len(pts):
            ax.scatter(pts[:, 1], pts[:, 0],
                       s=15, c=color, marker='x', linewidths=0.8,
                       label=f'{label} ({len(pts)})', zorder=3)
        ax.set_title(
            f'{label}\nprec={prec:.3f}  rec={rec:.3f}',
            fontsize=8)
        ax.legend(fontsize=7, loc='lower right')
        ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n',         type=int, default=20)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.3)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('finetuned.pt yükleniyor...')
    model = load_model(CKPT_FT, device)

    data    = np.load(NPZ_PATH, allow_pickle=True)
    n_total = int(data["n_images"][0])
    npz_dir = os.path.dirname(os.path.abspath(NPZ_PATH))
    records = []
    for i in range(n_total):
        raw = str(data[f"path_{i}"][0])
        if not os.path.isabs(raw):
            raw = os.path.join(npz_dir, raw)
        records.append({
            "path":     raw,
            "corners":  data[f"corners_{i}"],
            "img_size": tuple(data[f"img_size_{i}"]),
        })

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(len(records), size=min(args.n, len(records)), replace=False)

    print(f"\n{len(indices)} görüntü  —  GT=yeşil  OpenCV=sarı  CNN=camgöbeği\n")
    print(f"{'#':>3}  {'ocv prec':>9} {'ocv rec':>8}    {'cnn prec':>9} {'cnn rec':>8}  görüntü")
    print('-' * 75)

    op_list, or_list, cp_list, cr_list = [], [], [], []

    for rank, idx in enumerate(indices):
        r      = records[idx]
        img    = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'{rank+1:>3}  OKUNAMADI  {r["path"]}'); continue

        orig_w, orig_h = int(r["img_size"][0]), int(r["img_size"][1])
        img256 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # GT: npz'deki köşeler → 256×256 ölçekli (row, col)
        cx = r["corners"][:, 0] * (IMG_SIZE / orig_w)
        cy = r["corners"][:, 1] * (IMG_SIZE / orig_h)
        gt_rc = np.stack([cy, cx], axis=1)

        # OpenCV — 256×256 üzerinde
        ocv_xy = find_corners_opencv(img256)           # (x, y)
        if len(ocv_xy):
            ocv_rc = np.stack([ocv_xy[:, 1], ocv_xy[:, 0]], axis=1)
        else:
            ocv_rc = np.zeros((0, 2))

        # CNN
        hmap   = predict_heatmap(model, img256, device)
        cnn_rc = heatmap_to_corners(hmap, threshold=args.threshold)

        op, or_ = pr_score(ocv_rc, gt_rc)
        cp, cr  = pr_score(cnn_rc, gt_rc)
        op_list.append(op); or_list.append(or_)
        cp_list.append(cp); cr_list.append(cr)

        fname = os.path.basename(r["path"])
        print(f'{rank+1:>3}  {op:>9.3f} {or_:>8.3f}    {cp:>9.3f} {cr:>8.3f}  {fname}')

        fig = draw_comparison(
            img256, gt_rc, ocv_rc, cnn_rc,
            fname,
            op, or_, cp, cr)
        fig.savefig(os.path.join(OUT_DIR, f'{rank+1:02d}_{fname}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print('-' * 75)
    print(f"{'ORT':>3}  {np.mean(op_list):>9.3f} {np.mean(or_list):>8.3f}    "
          f"{np.mean(cp_list):>9.3f} {np.mean(cr_list):>8.3f}")
    print(f'\nGörseller: {OUT_DIR}/')


if __name__ == '__main__':
    main()
