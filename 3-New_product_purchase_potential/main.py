from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional, Dict

# 📦 Eğitimde kaydedilenleri yükle
try:
    model = tf.keras.models.load_model("category_multi_model.h5")
    scaler = joblib.load("scaler_multi.pkl")
    category_names = joblib.load("category_names.pkl")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model yüklenirken hata oluştu: {str(e)}")

# FastAPI app
app = FastAPI(
    title="Kategori Bazlı Ürün Öneri API",
    description="Müşterilerin kategori bazlı satın alma potansiyelini tahmin eden API",
    version="1.0"
)

# 🧾 Girdi formatı
class CustomerInput(BaseModel):
    Beverages: float = Field(0.0, ge=0, description="İçecekler kategorisi harcaması")
    Condiments: float = Field(0.0, ge=0, description="Baharatlar kategorisi harcaması")
    Confections: float = Field(0.0, ge=0, description="Şekerlemeler kategorisi harcaması")
    Dairy_Products: float = Field(0.0, ge=0, description="Süt Ürünleri kategorisi harcaması")
    Grains_Cereals: float = Field(0.0, ge=0, description="Tahıllar kategorisi harcaması")
    Meat_Poultry: float = Field(0.0, ge=0, description="Et/Kümes Hayvanları kategorisi harcaması")
    Produce: float = Field(0.0, ge=0, description="Sebze/Meyve kategorisi harcaması")
    Seafood: float = Field(0.0, ge=0, description="Deniz Ürünleri kategorisi harcaması")

    class Config:
        schema_extra = {
            "example": {
                "Beverages": 1000.0,
                "Condiments": 500.0,
                "Confections": 800.0,
                "Dairy_Products": 600.0,
                "Grains_Cereals": 400.0,
                "Meat_Poultry": 700.0,
                "Produce": 300.0,
                "Seafood": 900.0
            }
        }

# 🧠 Tahmin fonksiyonu
@app.post("/predict",
          summary="Kategori bazlı satın alma potansiyelini tahmin et",
          description="Müşterinin her bir kategori için satın alma potansiyelini tahmin eder",
          response_description="Kategori bazlı tahmin sonuçları")
def predict_category(data: CustomerInput):
    try:
        input_dict = data.dict()

        # Eksik kategorileri 0.0 ile tamamla
        for col in category_names:
            if col not in input_dict:
                input_dict[col] = 0.0

        # Doğru sırada DataFrame oluştur
        df = pd.DataFrame([input_dict])[category_names]

        # Ölçekle
        scaled = scaler.transform(df)

        # Tahmin et
        preds = model.predict(scaled)[0]
        results = {category: float(f"{prob:.2f}") for category, prob in zip(category_names, preds)}

        # En yüksek potansiyele sahip kategorileri belirle
        sorted_categories = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_categories = [{"kategori": cat, "olasilik": prob} for cat, prob in sorted_categories[:3]]

        return {
            "tahmin_edilen_kategori_olasiliklari": results,
            "oneri_kategorileri": top_categories,
            "toplam_harcama": sum(input_dict.values()),
            "en_yuksek_potansiyel": {
                "kategori": sorted_categories[0][0],
                "olasilik": float(f"{sorted_categories[0][1]:.2f}")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata oluştu: {str(e)}")

# Sağlık kontrolü endpoint'i
@app.get("/health", summary="API sağlık durumu", description="API'nin çalışır durumda olup olmadığını kontrol eder")
def health_check():
    return {"status": "healthy", "model_loaded": True}
