# checkerboard-corner-detector — Claude Bağlam Dosyası

Repo: `checkerboard-corner-detector` (public, açık kaynak)  
Ana proje: `heliumnitrate/opicaldistortionlab` → `OpticalDistortionLab/`

Bu repo, OpticalDistortionLab'ın CNN köşe dedektörünün bağımsız, herkese açık versiyonudur.

---

## Repo Yapısı

```
checkerboard-corner-detector/
├── data/
│   ├── synthesize.py       # Sentetik veri: checkerboard → distort → perspective → heatmap → augment
│   └── dataset.py          # PyTorch Dataset — on-the-fly, disk yok
├── model/
│   └── detector.py         # EfficientNet-V2-S encoder + U-Net decoder (21.92M param)
├── optical_distortion_engine/   # OpticalDistortionLab engine'inin subkümesi
│   ├── core/distortion_fun.py
│   └── estimation/
│       ├── checkerboard.py
│       ├── corner_detect.py
│       └── estimator.py
├── train.py                # Eğitim: focal BCE, AdamW, cosine LR, prec/rec metrik
├── infer.py                # Çıkarım: image → heatmap → NMS → köşe grid → distorsiyon
└── finetune.py             # Gerçek görüntü fine-tuning (50% real + 50% synthetic)
```

---

## CNN Pipeline

```
Eğitim (synthesize.py):
  make_checkerboard()
  → warp_checkerboard()       # backward mapping ile distortion uygula
  → _perspective_augment()    # %50 olasılıkla ≤10° X/Y tilt
  → make_heatmap()            # Gaussian blob (sigma=1.5 px)
  → _augment()                # blur, brightness, noise, JPEG

Loss: Focal BCE (gamma=2, alpha=0.5)
Optimizer: AdamW, lr=1e-4
Scheduler: CosineAnnealingLR
Metrik: Precision/Recall (tolerance=3 px, threshold=0.3)

Çıkarım (infer.py):
  image → predict_heatmap() → heatmap_to_corners() → sort_corners_to_grid()
        → estimate_from_corners() → c, d
```

---

## Fine-tuning (`finetune.py`)

- Girdi: `corners.npz` (gerçek köşe pozisyonları) + `model.pt`
- Çıktı: `finetuned.pt`
- Mix: %50 gerçek + %50 sentetik (catastrophic forgetting önleme)
- LR: 1e-5
- Resume from checkpoint destekleniyor

---

## Çalıştırma

```bash
# Sentetik eğitim
python train.py --epochs 50 --batch 8 --n_train 8000 --n_val 1000 --workers 4

# Gerçek görüntü fine-tuning
python finetune.py   # corners.npz ve model.pt gerekli

# Çıkarım
python infer.py
```

---

## docs/ Klasörü — Güncelleme Kuralları

Bu repodaki kod değişiklikleri **OpticalDistortionLab/docs/** dosyalarını da etkiler.
Her sohbetin sonunda, kod değişikliği yapıldıysa şu dosyaları güncelle:

```
OpticalDistortionLab/docs/
  PROJECT_SUMMARY.md   ← repo yapısı veya pipeline değiştiyse
  CHANGELOG.md         ← her önemli kod değişikliği için yeni madde ekle
  RESULTS.md           ← yeni eğitim / fine-tuning sonucu varsa
```

### Ne zaman güncellenir?

| Dosya | Tetikleyici |
|-------|------------|
| `PROJECT_SUMMARY.md` | Yeni dosya/fonksiyon eklendi, pipeline değişti |
| `CHANGELOG.md` | Yeni özellik, davranış değiştiren fix, yeni metrik |
| `RESULTS.md` | Yeni eğitim çalışması, fine-tuning epoch sonuçları |

### Ne zaman güncellenmez?

- Typo / yorum / isimlendirme düzeltmesi
- Tek seferlik config değişikliği
- Sadece test/inference çalıştırıldı, kod değişmedi

### Güncelleme formatı (CHANGELOG.md)

```
## #N — YYYY-MM | Başlık

**Eklenen / Değiştirilen dosyalar:**
- `dosya.py` — ne değişti

**Açıklama:**
Neden yapıldı, ne sorunu çözdü, varsa sayısal sonuç.
```
