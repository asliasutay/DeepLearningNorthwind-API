import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Model, scaler ve feature isimlerini yükle
model = tf.keras.models.load_model("first/siparis_model.h5")
scaler = joblib.load("first/scaler.pkl")
features = joblib.load("first/features.pkl")  # ÖNEMLİ!

# Yeni veri örneği (sadece bazı aylar aktif olabilir)
new_data = pd.DataFrame([{
    'total_orders': 3,
    'total_spent': 800.0,
    'avg_order_size': 266.6,
    'month_3.0': 1,
}])

# Eksik sütunları 0 ile tamamla
for col in features:
    if col not in new_data.columns:
        new_data[col] = 0

# Sıralamayı sabit tut
new_data = new_data[features]

# Ölçekle ve tahmin yap
new_data_scaled = scaler.transform(new_data)
prob = model.predict(new_data_scaled)[0][0]
prediction = int(prob > 0.5)

# Sonucu yazdır
print(f"Tahmin Olasılığı: {prob:.2f}")
print("Tahmin:", "Sipariş Verecek" if prediction == 1 else "Sipariş Vermeyecek")
