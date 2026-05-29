"""
main.py
-------
Konut Fiyat Tahmini — Merkezi Çalıştırma Sistemi

Kullanım:
    python main.py --mode all          # Tüm pipeline
    python main.py --mode preprocess   # Sadece veri ön işleme
    python main.py --mode train        # Sadece eğitim
    python main.py --mode evaluate     # Sadece değerlendirme
    python main.py --mode compare      # Adam vs SGD karşılaştırması
"""

import argparse
import sys
import os

# Proje kökünü Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import run_preprocessing
from src.train          import train, compare_optimizers
from src.evaluate       import run_evaluation
from src.experiments    import main as run_experiments


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║       KONUT FİYAT TAHMİN SİSTEMİ — MLP              ║
║       Yapay Sinir Ağları Dersi — Dönem Projesi        ║
╚══════════════════════════════════════════════════════╝
    """)


def run_all():
    """Tam pipeline: önişleme → eğitim → değerlendirme."""

    print("\n── ADIM 1: VERİ ÖN İŞLEME ──────────────────────────")
    run_preprocessing()

    print("\n── ADIM 2: MODEL EĞİTİMİ ────────────────────────────")
    train(
        model_name    = "mlp",
        epochs        = 150,
        batch_size    = 64,
        learning_rate = 1e-3,
        weight_decay  = 1e-4,
        optimizer_type= "adam",
        patience      = 20,
    )

    print("\n── ADIM 3: DEĞERLENDİRME ────────────────────────────")
    run_evaluation(model_name="mlp")

    print("\n✓ Pipeline tamamlandı. Sonuçlar: results/ klasörü")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Konut Fiyat Tahmini — MLP Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "preprocess", "train", "evaluate", "compare", "experiments"],
        help="Çalıştırılacak mod"
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all()

    elif args.mode == "preprocess":
        run_preprocessing()

    elif args.mode == "train":
        train(
            model_name    = "mlp",
            epochs        = 150,
            batch_size    = 64,
            learning_rate = 1e-3,
            weight_decay  = 1e-4,
            optimizer_type= "adam",
            patience      = 20,
        )

    elif args.mode == "evaluate":
        run_evaluation(model_name="mlp")

    elif args.mode == "compare":
        compare_optimizers()

    elif args.mode == "experiments":
        run_experiments()


if __name__ == "__main__":
    main()
