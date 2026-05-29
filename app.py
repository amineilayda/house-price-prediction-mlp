"""
app.py
------
Konut Fiyat Tahmini — Gradio Arayüzü

Çalıştırmak için:
    pip install gradio
    python app.py
"""

import os
import sys
import numpy as np
import joblib
import torch
import gradio as gr
from datetime import datetime

# Proje kökünü yola ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mlp_model import HousePriceMLP

# ─── Model ve Scaler Yükleme ───────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "mlp.pth")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "models", "y_scaler.pkl")

device = torch.device("cpu")

model = HousePriceMLP(input_size=24)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler   = joblib.load(SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

FEATURE_ORDER = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "lat", "long", "sqft_living15", "sqft_lot15",
    "sale_year", "sale_month", "house_age", "renovated",
    "years_since_reno", "sqft_per_bedroom", "sqft_living_ratio",
    "basement_ratio", "luxury_score",
]

# ─── Tahmin Fonksiyonu ─────────────────────────────────────────────────────────
def predict(
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, condition, grade, sqft_above, sqft_basement,
    yr_built, yr_renovated, lat, long_, sqft_living15, sqft_lot15,
):
    now = datetime.now()
    sale_year  = now.year
    sale_month = now.month

    # Feature engineering
    house_age        = sale_year - int(yr_built)
    renovated        = 1 if int(yr_renovated) > 0 else 0
    years_since_reno = (sale_year - int(yr_renovated)) if renovated else 0
    sqft_per_bedroom  = sqft_living / max(bedrooms, 1)
    sqft_living_ratio = sqft_living / max(sqft_living15, 1)
    basement_ratio    = sqft_basement / max(sqft_living, 1)
    luxury_score      = grade * 0.667 + view * 0.397 + waterfront * 0.266

    features = np.array([[
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        waterfront, view, condition, grade, sqft_above, sqft_basement,
        lat, long_, sqft_living15, sqft_lot15,
        sale_year, sale_month,
        house_age, renovated, years_since_reno,
        sqft_per_bedroom, sqft_living_ratio, basement_ratio, luxury_score,
    ]], dtype=np.float32)

    X_scaled = scaler.transform(features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    y_pred_log = y_scaler.inverse_transform(y_pred_scaled)
    price       = np.expm1(y_pred_log[0][0])

    return f"${price:,.0f}"


# ─── Gradio Arayüzü ────────────────────────────────────────────────────────────
with gr.Blocks(title="Konut Fiyat Tahmin", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🏠 Konut Fiyat Tahmini
        King County (Seattle) veri seti üzerinde eğitilmiş **MLP** modeli ile fiyat tahmini yapın.
        Bilgileri girin ve **Tahmin Et** butonuna tıklayın.
        """
    )

    with gr.Row():
        # Sol sütun
        with gr.Column():
            gr.Markdown("### 📐 Alan Bilgileri")
            sqft_living  = gr.Number(label="Yaşam Alanı (sqft)",    value=2000,  minimum=200)
            sqft_lot     = gr.Number(label="Arsa Alanı (sqft)",      value=5000,  minimum=500)
            sqft_above   = gr.Number(label="Bodrum Üstü Alan (sqft)",value=2000,  minimum=0)
            sqft_basement = gr.Number(label="Bodrum Alanı (sqft)",   value=0,     minimum=0)
            sqft_living15 = gr.Number(label="Komşu Ort. Yaşam Alanı (sqft)", value=1800, minimum=200)
            sqft_lot15    = gr.Number(label="Komşu Ort. Arsa Alanı (sqft)",  value=5000, minimum=500)

        # Orta sütun
        with gr.Column():
            gr.Markdown("### 🛏️ Temel Özellikler")
            bedrooms   = gr.Slider(label="Yatak Odası",  minimum=1, maximum=10, step=1, value=3)
            bathrooms  = gr.Slider(label="Banyo",         minimum=0.5, maximum=8, step=0.25, value=2)
            floors     = gr.Slider(label="Kat Sayısı",    minimum=1, maximum=3.5, step=0.5, value=1)
            condition  = gr.Slider(label="Durum (1-5)",   minimum=1, maximum=5,  step=1, value=3)
            grade      = gr.Slider(label="Yapı Kalitesi (1-13)", minimum=1, maximum=13, step=1, value=7)

            gr.Markdown("### ✨ Ekstra Özellikler")
            waterfront = gr.Radio(label="Deniz/Göl Cephesi", choices=[(  "Yok", 0), ("Var", 1)], value=0)
            view       = gr.Slider(label="Manzara Puanı (0-4)", minimum=0, maximum=4, step=1, value=0)

        # Sağ sütun
        with gr.Column():
            gr.Markdown("### 📍 Konum")
            lat  = gr.Number(label="Enlem (Latitude)",  value=47.5,  minimum=47.1,  maximum=47.8)
            long_ = gr.Number(label="Boylam (Longitude)", value=-122.2, minimum=-122.5, maximum=-121.3)

            gr.Markdown("### 🔨 Yapı Bilgisi")
            yr_built     = gr.Number(label="İnşaat Yılı",    value=1990, minimum=1900, maximum=2015)
            yr_renovated = gr.Number(label="Renovasyon Yılı (0 = yok)", value=0, minimum=0, maximum=2015)

    gr.Markdown("---")

    with gr.Row():
        btn = gr.Button("🔍 Tahmin Et", variant="primary", size="lg")

    with gr.Row():
        output = gr.Textbox(
            label="Tahmini Fiyat",
            placeholder="Sonuç burada görünecek...",
            interactive=False,
            text_align="center",
            container=True,
        )

    gr.Markdown(
        """
        > **Not:** Model King County (Seattle, WA) verileriyle eğitilmiştir.
        > Tahminler yalnızca bu bölge için geçerlidir. R² ≈ 0.88
        """
    )

    btn.click(
        fn=predict,
        inputs=[
            bedrooms, bathrooms, sqft_living, sqft_lot, floors,
            waterfront, view, condition, grade, sqft_above, sqft_basement,
            yr_built, yr_renovated, lat, long_, sqft_living15, sqft_lot15,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(share=False)
