from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from typing import Optional

# Model ve veri ön işleme araçlarının yüklenmesi
try:
    model = tf.keras.models.load_model("siparis_model.h5")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("features.pkl")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model yüklenirken hata oluştu: {str(e)}")

# FastAPI uygulanması
app = FastAPI(
    title="Sipariş Verme Alışkanlığı API",
    description="Müşterilerin sipariş verme alışkanlıklarını tahmin eden API",
    version="1.0"
)

# 🔸 Giriş modeli
class OrderFeatures(BaseModel):
    total_orders: int = Field(..., description="Toplam sipariş sayısı")
    total_spent: float = Field(..., description="Toplam harcama miktarı")
    avg_order_size: float = Field(..., description="Ortalama sipariş büyüklüğü")
    month_1_0: int = Field(0, alias="month_1.0", description="Ocak ayı sipariş durumu")
    month_2_0: int = Field(0, alias="month_2.0", description="Şubat ayı sipariş durumu")
    month_3_0: int = Field(0, alias="month_3.0", description="Mart ayı sipariş durumu")
    month_4_0: int = Field(0, alias="month_4.0", description="Nisan ayı sipariş durumu")
    month_5_0: int = Field(0, alias="month_5.0", description="Mayıs ayı sipariş durumu")
    month_6_0: int = Field(0, alias="month_6.0", description="Haziran ayı sipariş durumu")
    month_7_0: int = Field(0, alias="month_7.0", description="Temmuz ayı sipariş durumu")
    month_8_0: int = Field(0, alias="month_8.0", description="Ağustos ayı sipariş durumu")
    month_9_0: int = Field(0, alias="month_9.0", description="Eylül ayı sipariş durumu")
    month_10_0: int = Field(0, alias="month_10.0", description="Ekim ayı sipariş durumu")
    month_11_0: int = Field(0, alias="month_11.0", description="Kasım ayı sipariş durumu")
    month_12_0: int = Field(0, alias="month_12.0", description="Aralık ayı sipariş durumu")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "total_orders": 10,
                "total_spent": 5000.0,
                "avg_order_size": 500.0,
                "month_1.0": 1,
                "month_2.0": 0,
                "month_3.0": 1,
                "month_4.0": 0,
                "month_5.0": 1,
                "month_6.0": 0,
                "month_7.0": 1,
                "month_8.0": 0,
                "month_9.0": 1,
                "month_10.0": 0,
                "month_11.0": 1,
                "month_12.0": 0
            }
        }

# 🔮 Tahmin endpoint'i
@app.post("/predict", 
          summary="Sipariş verme olasılığını tahmin et",
          description="Müşterinin sipariş geçmişine göre önümüzdeki 6 ay içinde sipariş verme olasılığını tahmin eder",
          response_description="Tahmin sonuçları")
def predict_order(data: OrderFeatures):
    try:
        # Girdiyi sözlük formatında al (alias adlarıyla)
        input_dict = data.dict(by_alias=True)

        # Modelin beklediği özellik sırasıyla DataFrame oluştur
        df_input = pd.DataFrame([input_dict])[feature_names]

        # Ölçekleme
        scaled_input = scaler.transform(df_input)

        # Tahmin
        prob = model.predict(scaled_input)[0][0]
        tahmin = "Sipariş Verecek" if prob > 0.5 else "Sipariş Vermeyecek"

        return {
            "siparis_verme_olasiligi": float(f"{prob:.2f}"),
            "tahmin": tahmin,
            "model_guveni": "Yüksek" if abs(prob - 0.5) > 0.3 else "Orta" if abs(prob - 0.5) > 0.1 else "Düşük"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata oluştu: {str(e)}")

# Sağlık kontrolü endpoint'i
@app.get("/health", summary="API sağlık durumu", description="API'nin çalışır durumda olup olmadığını kontrol eder")
def health_check():
    return {"status": "healthy", "model_loaded": True}
