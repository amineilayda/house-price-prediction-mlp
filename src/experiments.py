"""
experiments.py
--------------
Ders konularıyla örtüşen karşılaştırmalı deneyler.

Deneylerin Ders Konusu Eşleştirmesi:
  Deney 1 — Aktivasyon Fonksiyonları  : Hafta 2-3 (YSA Model Yapısı)
  Deney 2 — Mimari (Katman Sayısı)    : Hafta 3   (Katman Yapısı)
  Deney 3 — Dropout / Overfitting     : Hafta 4   (Problemler ve Çözümleri)
  Deney 4 — Learning Rate             : Hafta 5   (GY Eğitim Yöntemi)
  Deney 5 — Optimizer (Adam vs SGD)   : Hafta 6   (Momentumlu BP)
  Deney 6 — Batch Size                : Hafta 7   (Ardışıl Mod vs Yığın Mod)

Kullanım:
  python src/experiments.py           # Tüm deneyler
  python src/experiments.py --exp 5   # Sadece Deney 5
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_dataloaders

RESULTS_DIR = "results/experiments"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI: Esnek MLP
# ══════════════════════════════════════════════════════════════════════════════
class FlexMLP(nn.Module):
    """Deney parametrelerine göre dinamik olarak oluşturulan MLP."""

    def __init__(self, input_size, hidden_dims, dropout_rates,
                 activation="relu", use_batchnorm=True):
        super().__init__()

        act_map = {
            "relu":    nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh":    nn.Tanh(),
        }
        assert activation in act_map, f"Bilinmeyen aktivasyon: {activation}"

        layers = []
        in_dim = input_size
        for out_dim, dr in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_map[activation])
            if dr > 0:
                layers.append(nn.Dropout(p=dr))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    if activation == "relu":
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    else:
                        nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI: Mini Eğitim Döngüsü
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(model, train_loader, val_loader,
                   lr=1e-3, epochs=60, patience=15,
                   optimizer_type="adam", momentum=0.9, weight_decay=1e-4):
    """
    Hızlı deney döngüsü. Eğitim/val MSE geçmişini ve süreyi döndürür.
    """
    criterion = nn.MSELoss()

    if optimizer_type == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.7, patience=10, min_lr=1e-6
    )

    best_val, best_state = float("inf"), None
    no_improve = 0
    train_hist, val_hist = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Eğitim
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_loader.dataset)

        # Doğrulama
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= len(val_loader.dataset)

        scheduler.step(v_loss)
        train_hist.append(t_loss)
        val_hist.append(v_loss)

        if v_loss < best_val - 1e-5:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # R² hesapla
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds.extend(model(xb.to(DEVICE)).cpu().numpy().flatten())
            trues.extend(yb.numpy().flatten())
    preds, trues = np.array(preds), np.array(trues)
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "best_val_mse": best_val,
        "r2":           r2,
        "train_hist":   train_hist,
        "val_hist":     val_hist,
        "epochs_run":   len(train_hist),
        "time_s":       time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DENEY 1 — Aktivasyon Fonksiyonu Karşılaştırması  (Hafta 2-3)
# ══════════════════════════════════════════════════════════════════════════════
def exp1_activation(train_loader, val_loader, input_size):
    """
    ReLU, Sigmoid ve Tanh aktivasyon fonksiyonlarını karşılaştırır.

    Akademik Referans:
      Glorot, X., Bordes, A., & Bengio, Y. (2011). "Deep Sparse Rectifier
      Neural Networks." AISTATS 2011, JMLR W&CP 15, pp. 315-323.
    """
    print("\n" + "═"*60)
    print("  DENEY 1: AKTİVASYON FONKSİYONU KARŞILAŞTIRMASI")
    print("  Ders Konusu: Hafta 2-3 — YSA Model Yapısı")
    print("═"*60)

    activations = ["relu", "sigmoid", "tanh"]
    results = {}

    for act in activations:
        print(f"  → {act.upper()} test ediliyor...", end="", flush=True)
        model = FlexMLP(input_size, [128, 64, 32], [0.3, 0.2, 0.0],
                        activation=act).to(DEVICE)
        res = run_experiment(model, train_loader, val_loader, epochs=60)
        results[act] = res
        print(f" Val MSE: {res['best_val_mse']:.4f}  R²: {res['r2']:.4f}  "
              f"({res['epochs_run']} epoch)")

    # Tablo
    _print_table("Aktivasyon Fonksiyonu", results,
                 {"ReLU":    "relu",
                  "Sigmoid": "sigmoid",
                  "Tanh":    "tanh"})

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deney 1: Aktivasyon Fonksiyonu Karşılaştırması\n"
                 "(Hafta 2-3: YSA Model Yapısı)", fontsize=13, fontweight="bold")

    colors = {"relu": "#2196F3", "sigmoid": "#F44336", "tanh": "#4CAF50"}
    for act, res in results.items():
        axes[0].plot(res["train_hist"], label=act.upper(), color=colors[act], linewidth=2)
        axes[1].plot(res["val_hist"],   label=act.upper(), color=colors[act], linewidth=2)

    for ax, title in zip(axes, ["Eğitim Kaybı", "Doğrulama Kaybı"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "exp1_activation.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DENEY 2 — Mimari Karşılaştırması (Hafta 3)
# ══════════════════════════════════════════════════════════════════════════════
def exp2_architecture(train_loader, val_loader, input_size):
    """
    Farklı derinliklerdeki MLP mimarilerini karşılaştırır.

    Akademik Referans:
      Hornik, K., Stinchcombe, M., & White, H. (1989). "Multilayer
      Feedforward Networks are Universal Approximators."
      Neural Networks, 2(5), 359-366.
    """
    print("\n" + "═"*60)
    print("  DENEY 2: MİMARİ KARŞILAŞTIRMASI (KATMAN SAYISI)")
    print("  Ders Konusu: Hafta 3 — YSA'da Katman Yapısı")
    print("═"*60)

    architectures = {
        "1 Gizli Katman (Baseline)": ([64],          [0.0]),
        "2 Gizli Katman":            ([128, 64],      [0.3, 0.0]),
        "3 Gizli Katman":            ([128, 64, 32],  [0.3, 0.2, 0.0]),
        "4 Gizli Katman":            ([256,128,64,32],[0.3, 0.2, 0.1, 0.0]),
    }
    results = {}

    for name, (dims, drops) in architectures.items():
        params = sum(p.numel() for p in
                     FlexMLP(input_size, dims, drops).parameters())
        print(f"  → {name} ({params:,} param)...", end="", flush=True)
        model = FlexMLP(input_size, dims, drops).to(DEVICE)
        res = run_experiment(model, train_loader, val_loader, epochs=60)
        results[name] = res
        print(f" R²: {res['r2']:.4f}  ({res['epochs_run']} epoch)")

    _print_table("Mimari", results,
                 {k: k for k in architectures})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deney 2: Mimari Karşılaştırması\n"
                 "(Hafta 3: YSA'da Katman Yapısı)", fontsize=13, fontweight="bold")
    palette = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]
    for (name, res), color in zip(results.items(), palette):
        lbl = name.split("(")[0].strip()
        axes[0].plot(res["train_hist"], label=lbl, color=color, linewidth=2)
        axes[1].plot(res["val_hist"],   label=lbl, color=color, linewidth=2)
    for ax, title in zip(axes, ["Eğitim Kaybı", "Doğrulama Kaybı"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "exp2_architecture.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DENEY 3 — Dropout / Overfitting Analizi (Hafta 4)
# ══════════════════════════════════════════════════════════════════════════════
def exp3_dropout(train_loader, val_loader, input_size):
    """
    Farklı dropout oranlarının overfitting üzerindeki etkisi.

    Akademik Referans:
      Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent
      Neural Networks from Overfitting."
      JMLR, 15(1), 1929-1958.
    """
    print("\n" + "═"*60)
    print("  DENEY 3: DROPOUT / OVERFİTTİNG ANALİZİ")
    print("  Ders Konusu: Hafta 4 — Problemler ve Çözümleri")
    print("═"*60)

    dropout_configs = {
        "Dropout Yok  (p=0.0)": [0.0, 0.0, 0.0],
        "Düşük        (p=0.1)": [0.1, 0.1, 0.0],
        "Orta         (p=0.3)": [0.3, 0.2, 0.0],
        "Yüksek       (p=0.5)": [0.5, 0.4, 0.0],
    }
    results = {}

    for name, drops in dropout_configs.items():
        print(f"  → {name}...", end="", flush=True)
        model = FlexMLP(input_size, [128, 64, 32], drops).to(DEVICE)
        res = run_experiment(model, train_loader, val_loader, epochs=60)
        # Overfitting gap: son epoch train/val farkı
        gap = res["val_hist"][-1] - res["train_hist"][-1]
        res["gap"] = gap
        results[name] = res
        print(f" R²: {res['r2']:.4f}  Gap: {gap:+.4f}")

    # Özel tablo: overfitting gap ekle
    print(f"\n  {'Konfigürasyon':30} {'Val MSE':>10} {'R²':>8} {'Train-Val Gap':>14}")
    print("  " + "-"*65)
    for name, res in results.items():
        print(f"  {name:30} {res['best_val_mse']:>10.4f} {res['r2']:>8.4f} "
              f"  {res['gap']:>+10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deney 3: Dropout / Overfitting Analizi\n"
                 "(Hafta 4: YSA ile İlgili Problemler ve Çözümleri)",
                 fontsize=13, fontweight="bold")
    palette = ["#F44336", "#FF9800", "#2196F3", "#4CAF50"]
    for (name, res), color in zip(results.items(), palette):
        lbl = name.strip()
        axes[0].plot(res["train_hist"], label=lbl, color=color, linewidth=2)
        axes[1].plot(res["val_hist"],   label=lbl, color=color, linewidth=2)
    for ax, title in zip(axes, ["Eğitim Kaybı", "Doğrulama Kaybı"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "exp3_dropout.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DENEY 4 — Learning Rate Analizi (Hafta 5)
# ══════════════════════════════════════════════════════════════════════════════
def exp4_learning_rate(train_loader, val_loader, input_size):
    """
    Farklı öğrenme hızlarının yakınsama hızı ve nihai performansa etkisi.

    Akademik Referans:
      Bengio, Y. (2012). "Practical Recommendations for Gradient-Based
      Training of Deep Architectures." arXiv:1206.5533.
      → Öğrenme hızı en kritik hiperparametredir.
    """
    print("\n" + "═"*60)
    print("  DENEY 4: ÖĞRENME HIZI (LEARNING RATE) ANALİZİ")
    print("  Ders Konusu: Hafta 5 — GY Eğitim Yöntemi")
    print("═"*60)

    lr_values = {
        "LR = 0.01  (Yüksek)": 1e-2,
        "LR = 0.001 (Standart)": 1e-3,
        "LR = 0.0001 (Düşük)": 1e-4,
        "LR = 0.00001 (Çok Düşük)": 1e-5,
    }
    results = {}

    for name, lr in lr_values.items():
        print(f"  → {name}...", end="", flush=True)
        model = FlexMLP(input_size, [128, 64, 32], [0.3, 0.2, 0.0]).to(DEVICE)
        res = run_experiment(model, train_loader, val_loader,
                             lr=lr, epochs=60, patience=15)
        results[name] = res
        print(f" R²: {res['r2']:.4f}  ({res['epochs_run']} epoch)")

    _print_table("Öğrenme Hızı", results,
                 {k: k for k in lr_values})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deney 4: Öğrenme Hızı Analizi\n"
                 "(Hafta 5: Geriye Yayınım Eğitim Yöntemi)",
                 fontsize=13, fontweight="bold")
    palette = ["#F44336", "#2196F3", "#4CAF50", "#9C27B0"]
    for (name, res), color in zip(results.items(), palette):
        lbl = name.split("(")[0].strip()
        axes[0].plot(res["train_hist"], label=lbl, color=color, linewidth=2)
        axes[1].plot(res["val_hist"],   label=lbl, color=color, linewidth=2)
    for ax, title in zip(axes, ["Eğitim Kaybı", "Doğrulama Kaybı"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "exp4_learning_rate.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DENEY 5 — Optimizer: Adam vs SGD+Momentum (Hafta 6)
# ══════════════════════════════════════════════════════════════════════════════
def exp5_optimizer(train_loader, val_loader, input_size):
    """
    Adam ve SGD+Momentum optimizerlarını karşılaştırır.

    Akademik Referanslar:
      Polyak, B. T. (1964). "Some methods of speeding up the convergence
      of iteration methods." USSR Comp. Math., 4(5), 1-17.
      → Momentumlu SGD (Hafta 6 konusu)

      Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic
      Optimization." ICLR 2015.
      → Adam = Momentum + RMSProp
    """
    print("\n" + "═"*60)
    print("  DENEY 5: OPTİMİZATÖR KARŞILAŞTIRMASI")
    print("  Ders Konusu: Hafta 6 — Momentumlu BP Yöntemi")
    print("═"*60)

    configs = {
        "Adam (lr=1e-3)":             ("adam", 1e-3, 0.9),
        "SGD+Momentum (lr=1e-2)":     ("sgd",  1e-2, 0.9),
        "SGD+Momentum (lr=1e-3)":     ("sgd",  1e-3, 0.9),
        "SGD+Momentum m=0.99":        ("sgd",  1e-2, 0.99),
    }
    results = {}

    for name, (opt_type, lr, mom) in configs.items():
        print(f"  → {name}...", end="", flush=True)
        model = FlexMLP(input_size, [128, 64, 32], [0.3, 0.2, 0.0]).to(DEVICE)
        res = run_experiment(model, train_loader, val_loader,
                             lr=lr, epochs=60, optimizer_type=opt_type,
                             momentum=mom)
        results[name] = res
        print(f" R²: {res['r2']:.4f}  ({res['epochs_run']} epoch, {res['time_s']:.1f}s)")

    _print_table("Optimizer", results, {k: k for k in configs})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deney 5: Optimizer Karşılaştırması\n"
                 "(Hafta 6: Momentumlu BP Yöntemi)",
                 fontsize=13, fontweight="bold")
    palette = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]
    for (name, res), color in zip(results.items(), palette):
        axes[0].plot(res["train_hist"], label=name, color=color, linewidth=2)
        axes[1].plot(res["val_hist"],   label=name, color=color, linewidth=2)
    for ax, title in zip(axes, ["Eğitim Kaybı", "Doğrulama Kaybı"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "exp5_optimizer.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DENEY 6 — Batch Size: Ardışıl vs Yığın Mod (Hafta 7)
# ══════════════════════════════════════════════════════════════════════════════
def exp6_batch_size(input_size):
    """
    Farklı batch size değerlerinin eğitim dinamiklerine etkisi.

    Ders Konusu — Hafta 7:
      Ardışıl Mod (Online): batch_size = 1   → her örnekte güncelleme
      Mini-batch Mod:       batch_size = 32/64 → standart
      Yığın Mod (Batch):    batch_size = tüm veri

    Akademik Referans:
      Goodfellow, I., Bengio, Y., & Courville, A. (2016).
      "Deep Learning." MIT Press. Chapter 8: Optimization.
      → Mini-batch, ardışıl ve tam yığın modlarının trade-off analizi.
    """
    print("\n" + "═"*60)
    print("  DENEY 6: BATCH SIZE — ARDIŞIL vs YIĞIN MOD")
    print("  Ders Konusu: Hafta 7 — Eğitme Yönteminde Farklı Modlar")
    print("═"*60)

    batch_sizes = {
        "Ardışıl Mod  (bs=1)":    1,
        "Mini-batch   (bs=32)":   32,
        "Mini-batch   (bs=64)":   64,
        "Mini-batch   (bs=256)":  256,
    }
    results = {}

    for name, bs in batch_sizes.items():
        print(f"  → {name}...", end="", flush=True)
        train_loader, val_loader, _, _ = get_dataloaders(batch_size=bs)
        # bs=1 (Ardışıl Mod): BatchNorm tek örnekten istatistik hesaplayamaz.
        # Bu teorik bir kısıttır — BatchNorm tanımsızdır (Ioffe & Szegedy, 2015).
        # Ardışıl modda BatchNorm devre dışı bırakılır.
        use_bn = (bs > 1)
        model = FlexMLP(input_size, [128, 64, 32], [0.3, 0.2, 0.0],
                        use_batchnorm=use_bn).to(DEVICE)
        res = run_experiment(model, train_loader, val_loader, epochs=40, patience=15)
        results[name] = res
        bn_note = "" if use_bn else " (BN yok)"
        print(f" R²: {res['r2']:.4f}  Süre: {res['time_s']:.1f}s{bn_note}")

    # Özel tablo: süre ekle
    print(f"\n  {'Mod':28} {'Val MSE':>10} {'R²':>8} {'Epoch':>7} {'Süre(s)':>9}")
    print("  " + "-"*65)
    for name, res in results.items():
        print(f"  {name:28} {res['best_val_mse']:>10.4f} {res['r2']:>8.4f} "
              f"{res['epochs_run']:>7} {res['time_s']:>9.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deney 6: Batch Size Analizi\n"
                 "(Hafta 7: Ardışıl Mod vs Yığın Mod)",
                 fontsize=13, fontweight="bold")
    palette = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]
    for (name, res), color in zip(results.items(), palette):
        lbl = name.strip()
        axes[0].plot(res["train_hist"], label=lbl, color=color, linewidth=2)
        axes[1].plot(res["val_hist"],   label=lbl, color=color, linewidth=2)
    for ax, title in zip(axes, ["Eğitim Kaybı", "Doğrulama Kaybı"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "exp6_batch_size.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════
def _print_table(title, results, label_map):
    print(f"\n  {'Konfigürasyon':35} {'Val MSE':>10} {'R²':>8} {'Epoch':>7}")
    print("  " + "-"*63)
    for display, key in label_map.items():
        res = results[key]
        best = " ★" if res["r2"] == max(r["r2"] for r in results.values()) else ""
        print(f"  {display:35} {res['best_val_mse']:>10.4f} "
              f"{res['r2']:>8.4f}{best}  ({res['epochs_run']} epoch)")


def _save(fig, filename):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Grafik kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA GİRİŞ NOKTASI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=0,
                        help="Çalıştırılacak deney numarası (0=hepsi)")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║  KARŞILAŞTIRMALI DENEYLER — YSA Ders Projesi" + " "*13 + "║")
    print("╚" + "═"*58 + "╝")
    print(f"  Cihaz: {DEVICE}  |  Sonuçlar: {RESULTS_DIR}/\n")

    # Veri loaderları (deney 6 kendi yükleyecek)
    train_loader, val_loader, _, input_size = get_dataloaders(batch_size=64)
    print(f"  Giriş boyutu: {input_size}\n")

    exp_map = {
        1: lambda: exp1_activation(train_loader, val_loader, input_size),
        2: lambda: exp2_architecture(train_loader, val_loader, input_size),
        3: lambda: exp3_dropout(train_loader, val_loader, input_size),
        4: lambda: exp4_learning_rate(train_loader, val_loader, input_size),
        5: lambda: exp5_optimizer(train_loader, val_loader, input_size),
        6: lambda: exp6_batch_size(input_size),
    }

    to_run = [args.exp] if args.exp in exp_map else list(exp_map.keys())

    for exp_num in to_run:
        exp_map[exp_num]()

    print("\n\n" + "═"*60)
    print("  TÜM DENEYLER TAMAMLANDI")
    print(f"  Grafikler: {RESULTS_DIR}/")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
