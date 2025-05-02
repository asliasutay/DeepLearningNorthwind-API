import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1️⃣ Model, scaler ve kategori adlarını yükle
model = tf.keras.models.load_model("category_multi_model.h5")
scaler = joblib.load("scaler_multi.pkl")
category_names = joblib.load("category_names.pkl")

# 2️⃣ Yeni müşteri örneği (geçmiş harcamaları)
new_customer = pd.DataFrame([{
    'Beverages': 200.0,
    'Condiments': 50.0,
    'Confections': 150.0,
    'Dairy Products': 0.0,
    'Grains/Cereals': 0.0,
    'Meat/Poultry': 100.0,
    'Produce': 0.0,
    'Seafood': 0.0
}])

# 3️⃣ Eksik kategorileri sıfırla ve sırala
for col in category_names:
    if col not in new_customer.columns:
        new_customer[col] = 0.0
new_customer = new_customer[category_names]

# 4️⃣ Ölçekle
new_scaled = scaler.transform(new_customer)

# 5️⃣ Tahmin et
probs = model.predict(new_scaled)[0]

# 6️⃣ Sonuçları yazdır
print("📊 Tahmini Kategori Alım Olasılıkları:")
for category, prob in zip(category_names, probs):
    print(f"{category}: {prob:.2f}")
