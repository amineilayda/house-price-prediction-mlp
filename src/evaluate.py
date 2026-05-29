"""
evaluate.py
-----------
Eğitilmiş MLP modelinin test seti üzerinde değerlendirilmesi.

Metrikler:
    MSE  — Mean Squared Error         (ortalama karesel hata)
    RMSE — Root Mean Squared Error     (kök ortalama karesel hata)
    MAE  — Mean Absolute Error         (ortalama mutlak hata)
    R²   — Coefficient of Determination (belirlilik katsayısı)
    MAPE — Mean Absolute Percentage Error

Akademik Referanslar:
─────────────────────────────────────────────────────────────────
[1] Regresyon Metrikleri:
    Chai, T., & Draxler, R. R. (2014).
    "Root mean square error (RMSE) or mean absolute error (MAE)?
    Arguments against avoiding RMSE in the literature."
    Geoscientific Model Development, 7(3), 1247–1250.

[2] R² Katsayısı:
    Cameron, A. C., & Windmeijer, F. A. (1997).
    "An R-squared measure of goodness of fit for some common
    nonlinear regression models."
    Journal of Econometrics, 77(2), 329–342.

[3] Residual Analizi:
    Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012).
    "Introduction to Linear Regression Analysis" (5th ed.).
    Wiley. Chapter 4: Model Adequacy Checking.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

import joblib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_model import HousePriceMLP
from src.dataset import get_dataloaders

RESULTS_DIR = "results"
MODELS_DIR  = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Tahminleri Topla ──────────────────────────────────────────────────────────
def get_predictions(model: nn.Module, loader) -> tuple:
    """Test setindeki tüm tahmin ve gerçek değerleri döndürür."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            preds   = model(X_batch).cpu().numpy().flatten()
            targets = y_batch.numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

    return np.array(all_preds), np.array(all_targets)


# ─── Metrik Hesapları ──────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Regresyon metriklerini hesaplar.

    Returns
    -------
    dict: mse, rmse, mae, r2, mape
    """
    residuals = y_true - y_pred

    mse  = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(residuals))
    r2   = 1 - (np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    mape = np.mean(np.abs(residuals / (y_true + 1e-8))) * 100   # yüzde

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# ─── Sonuçları Yazdır ──────────────────────────────────────────────────────────
def print_metrics(metrics: dict, model_name: str = "MLP"):
    print(f"\n{'='*50}")
    print(f"  TEST SONUÇLARI — {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  MSE  : {metrics['MSE']:>18,.2f}")
    print(f"  RMSE : {metrics['RMSE']:>18,.2f}")
    print(f"  MAE  : {metrics['MAE']:>18,.2f}")
    print(f"  R²   : {metrics['R2']:>18.4f}  (1.0 mükemmel)")
    print(f"  MAPE : {metrics['MAPE']:>17.2f}%")
    print(f"{'='*50}")

    # Model kalitesini yorumla
    if metrics["R2"] >= 0.90:
        yorum = "Mükemmel — yüksek açıklama gücü"
    elif metrics["R2"] >= 0.75:
        yorum = "İyi — kabul edilebilir"
    elif metrics["R2"] >= 0.60:
        yorum = "Orta — iyileştirme önerilebilir"
    else:
        yorum = "Zayıf — mimari veya veri gözden geçirilmeli"

    print(f"  Yorum: {yorum}\n")


# ─── Görselleştirmeler ─────────────────────────────────────────────────────────
def plot_results(y_true: np.ndarray, y_pred: np.ndarray,
                 metrics: dict, model_name: str = "mlp"):
    """
    4 grafikten oluşan değerlendirme panosu:
    1. Tahmin vs Gerçek (scatter)
    2. Residual dağılımı (histogram)
    3. Residual vs Tahmin
    4. Mutlak hata dağılımı
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Model Değerlendirmesi — MLP  |  R²={metrics['R2']:.4f}  "
        f"RMSE={metrics['RMSE']:,.0f}",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── 1. Tahmin vs Gerçek ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.6, color="#2196F3", edgecolors="white",
                linewidths=0.4, s=50, label="Tahminler")

    # Mükemmel tahmin doğrusu
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=2, label="Mükemmel tahmin")

    ax1.set_xlabel("Gerçek Fiyat", fontsize=11)
    ax1.set_ylabel("Tahmin Edilen Fiyat", fontsize=11)
    ax1.set_title("Tahmin vs Gerçek", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 2. Residual Histogramı ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=20, color="#4CAF50", edgecolor="white",
             alpha=0.85, density=True)
    ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Sıfır hata")

    # Normal dağılım eğrisi
    mu, sigma = residuals.mean(), residuals.std()
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    ax2.plot(x_norm, (1/(sigma * np.sqrt(2*np.pi))) *
             np.exp(-0.5 * ((x_norm - mu)/sigma)**2),
             "navy", linewidth=2, label=f"N({mu/1e6:.2f}M, {sigma/1e6:.2f}M)")

    ax2.set_xlabel("Residual (Gerçek − Tahmin)", fontsize=11)
    ax2.set_ylabel("Yoğunluk", fontsize=11)
    ax2.set_title("Residual Dağılımı", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. Residual vs Tahmin ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_pred, residuals, alpha=0.6, color="#FF9800",
                edgecolors="white", linewidths=0.4, s=50)
    ax3.axhline(0, color="red", linestyle="--", linewidth=2)

    # ±1 std bandı
    ax3.axhline( residuals.std(), color="gray", linestyle=":", linewidth=1.5,
                 label=f"±1σ = {residuals.std()/1e6:.2f}M")
    ax3.axhline(-residuals.std(), color="gray", linestyle=":", linewidth=1.5)

    ax3.set_xlabel("Tahmin Edilen Fiyat", fontsize=11)
    ax3.set_ylabel("Residual", fontsize=11)
    ax3.set_title("Residual vs Tahmin", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 4. Metrik Özet Tablosu ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    tablo_data = [
        ["Metrik", "Değer", "Birim"],
        ["MSE",    f"{metrics['MSE']:,.0f}",  "$²"],
        ["RMSE",   f"{metrics['RMSE']:,.0f}", "$"],
        ["MAE",    f"{metrics['MAE']:,.0f}",  "$"],
        ["R²",     f"{metrics['R2']:.4f}",    "—"],
        ["MAPE",   f"{metrics['MAPE']:.2f}%", "%"],
    ]

    tablo = ax4.table(cellText=tablo_data[1:], colLabels=tablo_data[0],
                      loc="center", cellLoc="center")
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(11)
    tablo.scale(1.2, 1.8)

    # Başlık satırını renklendir
    for j in range(3):
        tablo[(0, j)].set_facecolor("#1565C0")
        tablo[(0, j)].set_text_props(color="white", fontweight="bold")

    # R² satırını özellikle vurgula
    r2_row = 4
    color = "#C8E6C9" if metrics["R2"] >= 0.75 else "#FFCCBC"
    for j in range(3):
        tablo[(r2_row, j)].set_facecolor(color)

    ax4.set_title("Test Seti Metrikleri", fontsize=12, fontweight="bold", pad=20)

    path = os.path.join(RESULTS_DIR, f"evaluation_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Değerlendirme grafiği kaydedildi: {path}")


# ─── Sonuçları Dosyaya Kaydet ──────────────────────────────────────────────────
def save_metrics(metrics: dict, model_name: str = "mlp"):
    path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"{'='*40}\n")
        for k, v in metrics.items():
            f.write(f"  {k:6s}: {v:.6f}\n")
    print(f"[✓] Metrikler kaydedildi: {path}")


# ─── Ana Değerlendirme Fonksiyonu ──────────────────────────────────────────────
def run_evaluation(model_name: str = "mlp", batch_size: int = 32):
    print(f"\n{'='*50}")
    print(f"  DEĞERLENDİRME BAŞLIYOR — {model_name.upper()}")
    print(f"{'='*50}")

    # Checkpoint yükle
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    input_size = checkpoint["input_size"]
    model = HousePriceMLP(input_size=input_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[✓] Model yüklendi: {checkpoint_path}")

    # Test loader
    _, _, test_loader, _ = get_dataloaders(batch_size)

    # Tahminler (normalize edilmiş uzayda)
    y_pred_norm, y_true_norm = get_predictions(model, test_loader)

    # Inverse transform → log uzayından gerçek fiyata çevir
    # Pipeline: normalize(log(price)) → model → denormalize → exp → gerçek fiyat
    y_scaler_path = os.path.join(MODELS_DIR, "y_scaler.pkl")
    y_scaler = joblib.load(y_scaler_path)
    y_pred_log = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true_log = y_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
    y_pred = np.expm1(y_pred_log)   # exp(x) - 1  →  gerçek fiyat
    y_true = np.expm1(y_true_log)

    # Metrikler
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, model_name)

    # Grafikler
    plot_results(y_true, y_pred, metrics, model_name)

    # Kaydet
    save_metrics(metrics, model_name)

    return metrics, y_true, y_pred


if __name__ == "__main__":
    run_evaluation(model_name="mlp")
