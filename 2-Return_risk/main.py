from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import joblib
from typing import Optional

app = FastAPI(
    title="Ürün İade Riski API",
    description="Ürün siparişlerinin iade edilme riskini tahmin eden API",
    version="1.0"
)

# 🔹 Model ve scaler yükleniyor
try:
    model = tf.keras.models.load_model("product_model.h5")
    scaler = joblib.load("product_scaler.pkl")
    feature_names = ['discount', 'quantity', 'unit_price']
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model yüklenirken hata oluştu: {str(e)}")

# 🔸 Girdi veri modeli
class ProductInput(BaseModel):
    discount: float = Field(..., ge=0, le=1, description="İndirim oranı (0-1 arası)")
    quantity: int = Field(..., gt=0, description="Ürün miktarı")
    unit_price: float = Field(..., gt=0, description="Birim fiyat")

    class Config:
        schema_extra = {
            "example": {
                "discount": 0.15,
                "quantity": 5,
                "unit_price": 25.0
            }
        }

@app.post("/predict",
          summary="İade riskini tahmin et",
          description="Ürün siparişinin iade edilme riskini tahmin eder",
          response_description="Tahmin sonuçları")
def predict_return_risk(data: ProductInput):
    try:
        # Girdiyi vektörle
        input_array = np.array([[data.discount, data.quantity, data.unit_price]])
        
        # Ölçekle
        input_scaled = scaler.transform(input_array)
        
        # Tahmin
        prob = model.predict(input_scaled)[0][0]
        
        result = "İade riski yüksek" if prob > 0.5 else "İade riski düşük"

        return {
            "risk_olasiligi": float(f"{prob:.2f}"),
            "tahmin": result,
            "model_guveni": "Yüksek" if abs(prob - 0.5) > 0.3 else "Orta" if abs(prob - 0.5) > 0.1 else "Düşük",
            "risk_faktoru": {
                "indirim_etkisi": float(f"{data.discount * 100:.1f}%"),
                "miktar_etkisi": data.quantity,
                "fiyat_etkisi": float(f"{data.unit_price:.2f}")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata oluştu: {str(e)}")

# Sağlık kontrolü endpoint'i
@app.get("/health", summary="API sağlık durumu", description="API'nin çalışır durumda olup olmadığını kontrol eder")
def health_check():
    return {"status": "healthy", "model_loaded": True}
