import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sqlalchemy import create_engine
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# PostgreSQL bağlantısı
user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "database"
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

# Veriyi çek
query = """
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(od.quantity * od.unit_price) AS total_spent,
        AVG(od.quantity * od.unit_price) AS avg_order_size,
        MAX(o.order_date) AS last_order_date,
        EXTRACT(MONTH FROM MAX(o.order_date)) AS last_order_month
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN order_details od ON o.order_id = od.order_id
    GROUP BY c.customer_id
)
SELECT * FROM customer_metrics
"""
df = pd.read_sql(query, engine)

# Veri ön işleme fonksiyonu
def prepare_data(df):
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    current_date = df['last_order_date'].max()
    df['will_order'] = ((current_date - df['last_order_date']) <= timedelta(days=180)).astype(int)
    df = pd.get_dummies(df, columns=['last_order_month'], prefix='month')

    # NaN ve sonsuz değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

df_prepared = prepare_data(df)

# Mevsimsellik yakalanması
month_cols = [col for col in df_prepared.columns if col.startswith('month_')]
monthly_order_counts = df_prepared[month_cols].sum().sort_index()
month_labels = {
    'month_1.0': 'Ocak', 'month_2.0': 'Şubat', 'month_3.0': 'Mart', 'month_4.0': 'Nisan',
    'month_5.0': 'Mayıs', 'month_6.0': 'Haziran', 'month_7.0': 'Temmuz', 'month_8.0': 'Ağustos',
    'month_9.0': 'Eylül', 'month_10.0': 'Ekim', 'month_11.0': 'Kasım', 'month_12.0': 'Aralık'
}
monthly_order_counts.index = [month_labels.get(col, col) for col in monthly_order_counts.index]
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_order_counts.index, y=monthly_order_counts.values, hue=monthly_order_counts.index, palette='coolwarm', legend=False)
plt.title("Aylara Göre Sipariş Veren Müşteri Sayısı")
plt.xlabel("Ay")
plt.ylabel("Müşteri Sayısı")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Özellikler ve etiket
features = ['total_orders', 'total_spent', 'avg_order_size'] + [col for col in df_prepared.columns if col.startswith('month_')]
X = df_prepared[features]
y = df_prepared['will_order']

# Veriyi ayır ve ölçeklendir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dengesiz veri için ağırlık hesapla
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model tanımı
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
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

# Modelin eğitimi
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1,
    class_weight=class_weights
)

# Tahmin ve değerlendirme
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Eğitim grafiği
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

model.save("1-Ordering_habits/ordering_habits.h5")  # modeli kaydet
import joblib
joblib.dump(scaler, "1-Ordering_habits/scaler.pkl")  # scaler'ı kaydet
joblib.dump(features, "1-Ordering_habits/features.pkl")