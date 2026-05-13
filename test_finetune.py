"""
test_finetune.py
================
model.pt vs finetuned.pt karşılaştırması.

Test görüntüleri iki kaynaktan alınabilir:

  1. test_images/ klasörü (önerilen):
     - Görüntüler bilinen distorsiyon + gürültü ile warp edilir
     - GT köşeler OpenCV ile çıkarılır, sonra distort_corners() ile dönüştürülür
     - Gerçek test koşulu: model görmediği görüntüler, bilinen GT

  2. corners.npz (varsayılan):
     - Fine-tuning verisinden rastgele örnekler
     - Hızlı kontrol için

Kullanım:
    python test_finetune.py                        # corners.npz'den
    python test_finetune.py --source test_images   # test_images/ klasöründen
    python test_finetune.py --source test_images --distortion strong
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from optical_distortion_engine.estimation.checkerboard import distort_corners

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH    = os.path.join(REPO_DIR, '..', 'OpticalDistortionLab', 'datasets', 'corners.npz')
CKPT_BASE   = os.path.join(REPO_DIR, 'checkpoints', 'model.pt')
CKPT_FT     = os.path.join(REPO_DIR, 'checkpoints', 'finetuned.pt')
TEST_DIR    = os.path.join(REPO_DIR, 'test_images')
OUT_DIR     = os.path.join(REPO_DIR, 'test_results')
IMG_SIZE    = 256

BOARD_SIZES = [
    (7, 11), (11, 7), (7, 3), (3, 7),
    (9, 6),  (6, 9),  (8, 6), (6, 8),
    (7, 6),  (6, 7),
    # Chess boards (8×8 squares = 7×7 inner corners)
    (7, 7),
    # Calibration boards
    (12, 7), (7, 12),
    (10, 7), (7, 10),
    (9, 7),  (7, 9),
]

# Distorsiyon presetleri
DISTORTION_PRESETS = {
    'mild':   {'c': np.array([-0.15, 0.05]), 'd': np.array([3., 5.]), 'r_max': 0.8},
    'medium': {'c': np.array([-0.30, 0.12]), 'd': np.array([3., 5.]), 'r_max': 0.9},
    'strong': {'c': np.array([-0.50, 0.30]), 'd': np.array([3., 5.]), 'r_max': 1.0},
}


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def apply_distortion_and_noise(img_gray, c, d, r_max, noise_std=15):
    """Görüntüye barrel distorsiyon + Gaussian gürültü uygula."""
    from data.synthesize import warp_checkerboard
    H, W   = img_gray.shape
    center = ((H - 1) / 2.0, (W - 1) / 2.0)
    PSN    = r_max / (np.sqrt(2) * (max(H, W) - 1) / 2.0)

    distorted = warp_checkerboard(img_gray, c, d, center, PSN)
    noise     = np.random.normal(0, noise_std, distorted.shape).astype(np.int16)
    distorted = np.clip(distorted.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return distorted, center, PSN


def find_corners_opencv(img_gray):
    """OpenCV ile köşe bul, tahta boyutunu otomatik dene."""
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
            return True, corners, (rows, cols)
        ok, corners = cv2.findChessboardCorners(
            small, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(small, corners, (11, 11), (-1, -1), crit)
            corners = corners.reshape(-1, 2) * scale
            return True, corners, (rows, cols)
    return False, None, None


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


def scale_corners_xy_to_rc(corners_xy, orig_w, orig_h, S=256):
    cx = corners_xy[:, 0] * (S / orig_w)
    cy = corners_xy[:, 1] * (S / orig_h)
    return np.stack([cy, cx], axis=1)


def draw_result(img, gt_rc, pred_base, pred_ft, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=9)
    for ax, pts, label, color in zip(
        axes,
        [pred_base, pred_ft, gt_rc],
        [f'model.pt ({len(pred_base)})', f'finetuned.pt ({len(pred_ft)})', f'GT ({len(gt_rc)})'],
        ['red', 'cyan', 'lime']
    ):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        if len(pts):
            ax.scatter(pts[:, 1], pts[:, 0], s=10, c=color, linewidths=0)
        ax.set_title(label, fontsize=9); ax.axis('off')
    plt.tight_layout()
    return fig


# ─── Kaynak 1: test_images/ klasörü ──────────────────────────────────────────

def test_from_folder(model_base, model_ft, args, device):
    os.makedirs(TEST_DIR, exist_ok=True)
    exts   = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [p for p in sorted(os.listdir(TEST_DIR))
              if os.path.splitext(p)[1].lower() in exts]

    if not images:
        print(f"test_images/ klasörü boş veya yok: {TEST_DIR}")
        print("Görüntüleri buraya koy ve tekrar çalıştır.")
        return

    preset = DISTORTION_PRESETS[args.distortion]
    print(f"Distorsiyon: {args.distortion}  c={preset['c']}  noise_std={args.noise_std}")
    print(f"\n{'#':>3}  {'base prec':>10} {'base rec':>9}  {'ft prec':>9} {'ft rec':>8}  görüntü")
    print('-' * 75)

    bp_list, br_list, fp_list, fr_list = [], [], [], []

    for rank, fname in enumerate(images):
        path = os.path.join(TEST_DIR, fname)
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'{rank+1:>3}  OKUNAMADI  {fname}')
            continue

        # OpenCV ile undistorted görüntüde köşeleri bul
        ok, corners_xy, board = find_corners_opencv(img)
        if not ok:
            print(f'{rank+1:>3}  köşe bulunamadı (OpenCV)  {fname}')
            continue

        orig_h, orig_w = img.shape

        # Distorsiyon + gürültü uygula
        dist_img, center, PSN = apply_distortion_and_noise(
            img, preset['c'], preset['d'], preset['r_max'], args.noise_std)

        # GT köşeleri de dönüştür
        corners_rc = np.stack([corners_xy[:, 1], corners_xy[:, 0]], axis=1)  # (N,2) row,col
        corners_grid = corners_rc.reshape(-1, 1, 2)
        gt_dist_grid = distort_corners(corners_grid, center, PSN, preset['c'], preset['d'])
        gt_dist_rc   = gt_dist_grid.reshape(-1, 2)

        # 256×256'ya resize
        img256  = cv2.resize(dist_img, (IMG_SIZE, IMG_SIZE))
        scale_r = IMG_SIZE / orig_h
        scale_c = IMG_SIZE / orig_w
        gt_256  = gt_dist_rc * np.array([scale_r, scale_c])

        # Tahminler
        hmap_base = predict_heatmap(model_base, img256, device)
        hmap_ft   = predict_heatmap(model_ft,   img256, device)
        pred_base = heatmap_to_corners(hmap_base, threshold=args.threshold)
        pred_ft   = heatmap_to_corners(hmap_ft,   threshold=args.threshold)

        bp, br = pr_score(pred_base, gt_256)
        fp, fr = pr_score(pred_ft,   gt_256)
        bp_list.append(bp); br_list.append(br)
        fp_list.append(fp); fr_list.append(fr)
        print(f'{rank+1:>3}  {bp:>10.3f} {br:>9.3f}  {fp:>9.3f} {fr:>8.3f}  {fname}')

        # Görsel
        fig = draw_result(img256, gt_256, pred_base, pred_ft,
                          f'{fname}  [{args.distortion}+noise{args.noise_std}]')
        fig.savefig(os.path.join(OUT_DIR, f'{rank+1:02d}_{fname}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    if bp_list:
        print('-' * 75)
        print(f"{'ORT':>3}  {np.mean(bp_list):>10.3f} {np.mean(br_list):>9.3f}  "
              f"{np.mean(fp_list):>9.3f} {np.mean(fr_list):>8.3f}")


# ─── Kaynak 2: corners.npz ───────────────────────────────────────────────────

def test_from_npz(model_base, model_ft, args, device):
    data    = np.load(NPZ_PATH, allow_pickle=True)
    n       = int(data["n_images"][0])
    npz_dir = os.path.dirname(os.path.abspath(NPZ_PATH))
    records = []
    for i in range(n):
        raw = str(data[f"path_{i}"][0])
        if not os.path.isabs(raw):
            raw = os.path.join(npz_dir, raw)
        records.append({"path": raw, "corners": data[f"corners_{i}"],
                        "img_size": tuple(data[f"img_size_{i}"])})

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(len(records), size=min(args.n, len(records)), replace=False)

    print(f"corners.npz'den {len(indices)} görüntü  (threshold={args.threshold})\n")
    print(f"{'#':>3}  {'base prec':>10} {'base rec':>9}  {'ft prec':>9} {'ft rec':>8}  görüntü")
    print('-' * 75)

    bp_list, br_list, fp_list, fr_list = [], [], [], []

    for rank, idx in enumerate(indices):
        r     = records[idx]
        img   = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'{rank+1:>3}  OKUNAMADI'); continue

        orig_w, orig_h = int(r["img_size"][0]), int(r["img_size"][1])
        img256 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gt_rc  = scale_corners_xy_to_rc(r["corners"], orig_w, orig_h)

        hmap_base = predict_heatmap(model_base, img256, device)
        hmap_ft   = predict_heatmap(model_ft,   img256, device)
        pred_base = heatmap_to_corners(hmap_base, threshold=args.threshold)
        pred_ft   = heatmap_to_corners(hmap_ft,   threshold=args.threshold)

        bp, br = pr_score(pred_base, gt_rc)
        fp, fr = pr_score(pred_ft,   gt_rc)
        bp_list.append(bp); br_list.append(br)
        fp_list.append(fp); fr_list.append(fr)
        print(f'{rank+1:>3}  {bp:>10.3f} {br:>9.3f}  {fp:>9.3f} {fr:>8.3f}  '
              f'{os.path.basename(r["path"])}')

        fig = draw_result(img256, gt_rc, pred_base, pred_ft,
                          os.path.basename(r["path"]))
        fig.savefig(os.path.join(OUT_DIR, f'{rank+1:02d}_{os.path.basename(r["path"])}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print('-' * 75)
    print(f"{'ORT':>3}  {np.mean(bp_list):>10.3f} {np.mean(br_list):>9.3f}  "
          f"{np.mean(fp_list):>9.3f} {np.mean(fr_list):>8.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',     choices=['npz', 'test_images'], default='test_images')
    p.add_argument('--distortion', choices=['mild', 'medium', 'strong'], default='strong')
    p.add_argument('--noise_std',  type=float, default=20)
    p.add_argument('--threshold',  type=float, default=0.3)
    p.add_argument('--n',          type=int,   default=8)
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('model.pt yükleniyor...')
    model_base = load_model(CKPT_BASE, device)
    print('finetuned.pt yükleniyor...')
    model_ft   = load_model(CKPT_FT,   device)

    if args.source == 'test_images':
        test_from_folder(model_base, model_ft, args, device)
    else:
        test_from_npz(model_base, model_ft, args, device)

    print(f'\nGörseller: {OUT_DIR}/')


if __name__ == '__main__':
    main()
