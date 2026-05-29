"""
train.py
--------
Konut Fiyat Tahmini — MLP Eğitim Modülü

Akademik Referanslar:
─────────────────────────────────────────────────────────────────
[1] Geri Yayınım (Backpropagation):
    Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
    "Learning representations by back-propagating errors."
    Nature, 323(6088), 533–536.

[2] Adam Optimizer:
    Kingma, D. P., & Ba, J. (2015).
    "Adam: A Method for Stochastic Optimization."
    ICLR 2015. arXiv:1412.6980.
    → Momentum + RMSProp'u birleştirir. Adaptif öğrenme hızı.
    → SGD+Momentum'dan genellikle daha hızlı yakınsama.

[3] SGD + Momentum (karşılaştırma için):
    Polyak, B. T. (1964).
    "Some methods of speeding up the convergence of iteration methods."
    USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17.
    → Dersin 6. haftası: Momentumlu BP yöntemi

[4] Erken Durdurma (Early Stopping):
    Prechelt, L. (1998).
    "Early Stopping - But When?"
    In Neural Networks: Tricks of the Trade, pp. 55–69. Springer.
    → Validation kaybı iyileşmediğinde eğitimi durdurur, overfitting önler.

[5] Öğrenme Hızı Zamanlayıcı (ReduceLROnPlateau):
    Bengio, Y. (2012).
    "Practical Recommendations for Gradient-Based Training of Deep
    Architectures." arXiv:1206.5533.
    → Plato'ya takılan modeli küçük lr ile kurtarır.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_model import HousePriceMLP
from src.dataset import get_dataloaders

# ─── Sabitler ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
MODELS_DIR  = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Erken Durdurma ────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Validation kaybı 'patience' epoch boyunca iyileşmezse eğitimi durdurur.
    En iyi model ağırlıklarını saklar.

    Kaynak: Prechelt (1998)
    """
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # En iyi ağırlıkları kopyala
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience  # True → eğitimi durdur

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ─── Tek Epoch Eğitimi ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer) -> float:
    """
    Batch Modu eğitim — Dersin 7. haftası: Yığın Mod (Batch Mode).
    Her batch sonrası ağırlıklar güncellenir.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        # 1. İleri geçiş (Forward pass)
        predictions = model(X_batch)

        # 2. Kayıp hesapla
        loss = criterion(predictions, y_batch)

        # 3. Gradyanları sıfırla
        optimizer.zero_grad()

        # 4. Geri yayınım (Backpropagation) [Rumelhart et al., 1986]
        loss.backward()

        # 5. Ağırlıkları güncelle [Kingma & Ba, 2015]
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# ─── Doğrulama ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# ─── Loss Grafiği ──────────────────────────────────────────────────────────────
def plot_loss(train_losses, val_losses, model_name: str = "mlp"):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_losses, label="Eğitim Kaybı",    color="#2196F3", linewidth=2)
    ax.plot(val_losses,   label="Doğrulama Kaybı", color="#F44336", linewidth=2, linestyle="--")

    best_epoch = val_losses.index(min(val_losses))
    ax.axvline(x=best_epoch, color="green", linestyle=":", linewidth=1.5,
               label=f"En iyi model (epoch {best_epoch+1})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Kaybı", fontsize=12)
    ax.set_title(f"Eğitim ve Doğrulama Kaybı — {model_name.upper()}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"loss_graph_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Loss grafiği kaydedildi: {path}")


# ─── Ana Eğitim Fonksiyonu ─────────────────────────────────────────────────────
def train(
    model_name:   str   = "mlp",
    epochs:       int   = 200,
    batch_size:   int   = 32,
    learning_rate:float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 25,
    optimizer_type: str = "adam",   # "adam" veya "sgd"
    momentum:     float = 0.9,      # SGD için [Polyak, 1964]
):
    """
    Parameters
    ----------
    model_name     : Kaydedilecek dosya adı
    epochs         : Maksimum epoch sayısı
    batch_size     : Mini-batch boyutu
    learning_rate  : Başlangıç öğrenme hızı
    weight_decay   : L2 düzenileştirme katsayısı
    patience       : Early stopping sabrı
    optimizer_type : 'adam' veya 'sgd'
    momentum       : SGD momentum katsayısı [Polyak, 1964]
    """

    print(f"\n{'='*55}")
    print(f"  EĞİTİM BAŞLIYOR — {model_name.upper()}")
    print(f"{'='*55}")
    print(f"  Cihaz        : {DEVICE}")
    print(f"  Optimizer    : {optimizer_type.upper()}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Max Epoch    : {epochs}")
    print(f"  Early Stop   : {patience} epoch\n")

    # Veri yükle
    train_loader, val_loader, _, input_size = get_dataloaders(batch_size)

    # Model oluştur
    model = HousePriceMLP(input_size=input_size).to(DEVICE)
    model.summary()

    # Kayıp fonksiyonu — MSE (Mean Squared Error)
    criterion = nn.MSELoss()

    # Optimizer seçimi
    if optimizer_type == "adam":
        # Adam [Kingma & Ba, 2015]
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        # SGD + Momentum [Polyak, 1964] — Dersin 6. haftası
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )

    # Öğrenme hızı zamanlayıcı [Bengio, 2012]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=20, min_lr=1e-6
    )

    # Erken durdurma [Prechelt, 1998]
    early_stopping = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss   = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Her 10 epochta bir yazdır
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"Train: {train_loss:.4e} | "
                  f"Val: {val_loss:.4e} | "
                  f"LR: {lr_now:.2e}")

        # Erken durdurma kontrolü
        if early_stopping(val_loss, model):
            print(f"\n  ⏹  Early stopping: {epoch}. epochta duruldu.")
            break

    # En iyi ağırlıkları geri yükle
    early_stopping.restore_best(model)

    elapsed = time.time() - start_time
    print(f"\n  Eğitim süresi : {elapsed:.1f}s")
    print(f"  En iyi val MSE: {early_stopping.best_loss:.4e}")

    # Modeli kaydet
    save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size":       input_size,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "best_val_loss":    early_stopping.best_loss,
    }, save_path)
    print(f"  Model kaydedildi: {save_path}")

    # Loss grafiği
    plot_loss(train_losses, val_losses, model_name)

    return model, train_losses, val_losses


# ─── Adam vs SGD Karşılaştırması ───────────────────────────────────────────────
def compare_optimizers():
    """
    Adam ve SGD+Momentum'u karşılaştırır.
    Akademik rapor için önemli bir deney.
    """
    print("\n" + "="*55)
    print("  OPTİMİZATÖR KARŞILAŞTIRMASI")
    print("="*55)

    results = {}

    for opt in ["adam", "sgd"]:
        print(f"\n→ {opt.upper()} ile eğitim...")
        _, train_l, val_l = train(
            model_name    = f"model_{opt}",
            epochs        = 150,
            batch_size    = 64,
            weight_decay  = 1e-4,
            optimizer_type= opt,
            learning_rate = 1e-3 if opt == "adam" else 1e-2,
            patience      = 20,
        )
        results[opt] = {"best_val": min(val_l), "epochs": len(val_l)}

    print("\n" + "="*55)
    print("  KARŞILAŞTIRMA SONUÇLARI")
    print("="*55)
    for opt, res in results.items():
        print(f"  {opt.upper():5s} | Best Val MSE: {res['best_val']:.4e} | Epochs: {res['epochs']}")
    print("="*55)


# ─── Giriş Noktası ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, train_losses, val_losses = train(
        model_name    = "mlp",
        epochs        = 150,
        batch_size    = 64,
        learning_rate = 1e-3,
        weight_decay  = 1e-4,
        optimizer_type= "adam",
        patience      = 20,
    )
