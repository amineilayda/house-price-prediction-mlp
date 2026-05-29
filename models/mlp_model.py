"""
mlp_model.py
------------
Konut Fiyat Tahmini için Çok Katmanlı Algılayıcı (MLP) Modeli

Mimari Kararları ve Akademik Kaynaklar:
─────────────────────────────────────────────────────────────────
[1] Aktivasyon - ReLU (Rectified Linear Unit):
    Glorot, X., Bordes, A., & Bengio, Y. (2011).
    "Deep Sparse Rectifier Neural Networks."
    AISTATS 2011. JMLR W&CP 15, pp. 315–323.
    → Sigmoid/tanh'a kıyasla vanishing gradient sorununu azaltır.
    → Seyrek aktivasyon ile daha iyi genelleme sağlar.

[2] Toplu Normalleştirme (Batch Normalization):
    Ioffe, S., & Szegedy, C. (2015).
    "Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift."
    ICML 2015. PMLR 37, pp. 448–456.
    → Her katmanın girişini normalize ederek eğitimi hızlandırır.
    → Yüksek öğrenme hızlarına izin verir, düzenleyici etki gösterir.

[3] Düzenlileştirme - Dropout:
    Srivastava, N., Hinton, G., Krizhevsky, A., Sridharan, R., &
    Salakhutdinov, R. (2014).
    "Dropout: A Simple Way to Prevent Neural Networks from
    Overfitting."
    Journal of Machine Learning Research, 15(1), 1929–1958.
    → Eğitim sırasında rastgele nöronları devre dışı bırakır.
    → Aşırı öğrenmeyi (overfitting) engeller.

[4] Ağırlık Başlatma - He (Kaiming) Initialization:
    He, K., Zhang, X., Ren, S., & Sun, J. (2015).
    "Delving Deep into Rectifiers: Surpassing Human-Level
    Performance on ImageNet Classification."
    ICCV 2015, pp. 1026–1034.
    → ReLU aktivasyonu için optimal başlangıç ağırlıklarını sağlar.
    → Derin ağlarda gradyan akışını dengeler.

[5] Evrensel Yaklaşım Teoremi (Mimari Gerekçe):
    Hornik, K., Stinchcombe, M., & White, H. (1989).
    "Multilayer Feedforward Networks are Universal Approximators."
    Neural Networks, 2(5), 359–366.
    → Yeterli nöronla MLP herhangi bir sürekli fonksiyonu yaklaşabilir.

Mimari:
    Giriş(24) → [FC(256) → BN → ReLU → Drop(0.3)]
              → [FC(128) → BN → ReLU → Drop(0.2)]
              → [FC(64)  → BN → ReLU → Drop(0.1)]
              → [FC(32)  → BN → ReLU]
              → Çıkış(1)  [Doğrusal — Regresyon]

    King County veri seti (21,613 örnek) için derin mimari.
    21K örnek yeterli kapasiteyle öğrenmeye izin verir.
"""

import torch
import torch.nn as nn


class HousePriceMLP(nn.Module):
    """
    Konut fiyatı regresyonu için Çok Katmanlı Algılayıcı.

    Parameters
    ----------
    input_size  : int   — Giriş özellik sayısı (varsayılan: 24)
    hidden_dims : list  — Gizli katman nöron sayıları
    dropout_rates: list — Her gizli katmana ait dropout oranları
    """

    def __init__(
        self,
        input_size:    int  = 24,
        hidden_dims:   list = [256, 128, 64, 32],
        dropout_rates: list = [0.3, 0.2, 0.1, 0.0],
    ):
        super(HousePriceMLP, self).__init__()

        assert len(dropout_rates) == len(hidden_dims), \
            "hidden_dims ve dropout_rates aynı uzunlukta olmalıdır."

        layers = []
        in_dim = input_size

        for out_dim, drop_rate in zip(hidden_dims, dropout_rates):
            # Tam bağlantılı katman
            layers.append(nn.Linear(in_dim, out_dim))

            # Batch Normalization [Ioffe & Szegedy, 2015]
            layers.append(nn.BatchNorm1d(out_dim))

            # ReLU Aktivasyonu [Glorot et al., 2011]
            layers.append(nn.ReLU())

            # Dropout [Srivastava et al., 2014]
            if drop_rate > 0.0:
                layers.append(nn.Dropout(p=drop_rate))

            in_dim = out_dim

        # Çıkış katmanı — regresyon için doğrusal
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

        # He (Kaiming) Initialization [He et al., 2015]
        self._initialize_weights()

    def _initialize_weights(self):
        """
        ReLU aktivasyonu için He (Kaiming) normal başlatma uygular.
        Çıkış katmanında Xavier başlatma kullanılır (doğrusal aktivasyon).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Son katmanı ayır
                if module.out_features == 1:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # He initialization: std = sqrt(2 / fan_in)
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        """Eğitilebilir parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Model mimarisi ve parametre özetini yazdırır."""
        print("\n" + "=" * 55)
        print("         KONUT FİYAT TAHMİN MODELİ (MLP)")
        print("=" * 55)
        print(self.network)
        print("=" * 55)
        print(f"  Toplam Eğitilebilir Parametre: {self.count_parameters():,}")
        print("=" * 55 + "\n")


# ─── Karşılaştırma için Basit Temel Model ──────────────────────────────────────
class BaselineMLP(nn.Module):
    """
    Tek gizli katmanlı temel model.
    Akademik çalışmalarda karşılaştırma (baseline) için kullanılır.

    Referans:
        Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
        "Learning representations by back-propagating errors."
        Nature, 323(6088), 533–536.
    """

    def __init__(self, input_size: int = 14, hidden_size: int = 32):
        super(BaselineMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─── Hızlı Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = HousePriceMLP(input_size=24)
    model.summary()

    x_test = torch.randn(32, 24)
    model.eval()
    with torch.no_grad():
        out = model(x_test)

    print(f"İleri geçiş testi:")
    print(f"  Giriş : {x_test.shape}")
    print(f"  Çıkış : {out.shape}")
    print(f"  Parametre sayısı: {model.count_parameters():,}")
