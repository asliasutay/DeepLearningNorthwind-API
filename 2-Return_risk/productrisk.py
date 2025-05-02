import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# PostgreSQL bağlantısı
engine = create_engine("postgresql://postgres:12345@localhost:5432/database")

# Veriyi çek
query = """
    SELECT 
        o.customer_id,
        o.order_id,
        od.discount,
        od.quantity,
        od.unit_price,
        -- Sahte etiketleme: Yüksek indirim + düşük tutar iade riski demek
        CASE 
            WHEN od.discount > 0.2 AND (od.quantity * od.unit_price) < 100 THEN 1
            ELSE 0
        END as is_returned
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
"""
df = pd.read_sql(query, engine)

# Özellik seçimi
features = ['discount', 'quantity', 'unit_price']
X = df[features]
y = df['is_returned']

# Veriyi ayır ve ölçekle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weight (dengesiz veri için)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# Model tanımı
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model(X_train_scaled.shape[1])

# Eğit
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1,
    class_weight=class_weights
)

# Tahmin ve rapor
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Eğitim eğrileri
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title("Kayıp (Loss)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title("Doğruluk (Accuracy)")
plt.legend()

plt.tight_layout()
plt.show()

# 📦 Model ve scaler kaydı (API için gerekli)
model.save("second/product_model.h5")
joblib.dump(scaler, "second/product_scaler.pkl")
