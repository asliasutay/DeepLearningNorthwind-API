import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# 1️⃣ Veritabanı bağlantısı
engine = create_engine("postgresql://postgres:Yalova8988@localhost:5432/GYK1")

# 2️⃣ SQL ile veri çek
query = """
    SELECT 
        c.customer_id,
        o.order_id,
        od.product_id,
        p.product_name,
        cat.category_name,
        od.unit_price,
        od.quantity,
        (od.unit_price * od.quantity) AS total_price
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
"""
df = pd.read_sql(query, engine)

# 3️⃣ Kategoriye göre müşteri harcaması
df['total_price'] = df['unit_price'] * df['quantity']
category_spend = df.pivot_table(index='customer_id', columns='category_name', values='total_price', aggfunc='sum').fillna(0)

# 4️⃣ Giriş ve etiket verisi: Y harcamayı 0/1 yap
X = category_spend.copy()
y = (category_spend > 0).astype(int)  # Alışveriş yaptıysa 1, yapmadıysa 0

# 5️⃣ Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='sigmoid')  # çoklu çıktı
])
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 8️⃣ Eğit
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# 9️⃣ Kaydet
model.save("category_multi_model.h5")
joblib.dump(scaler, "scaler_multi.pkl")
joblib.dump(X.columns.tolist(), "category_names.pkl")
print("✅ Model ve araçlar kaydedildi.")
