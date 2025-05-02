import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Veri hazırlık ve augmentation fonksiyonları
def prepare_data(df):
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    current_date = df['last_order_date'].max()
    df['will_order'] = ((current_date - df['last_order_date']) <= pd.Timedelta(days=180)).astype(int)
    df = pd.get_dummies(df, columns=['last_order_month'], prefix='month')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def augment_customer_data(df, n_times=5, noise_level=0.1):
    augmented = []
    numeric_cols = ['total_orders', 'total_spent', 'avg_order_size']
    for _ in range(n_times):
        df_aug = df.copy()
        for col in numeric_cols:
            noise = np.random.normal(0, noise_level, size=len(df))
            df_aug[col] = df[col] * (1 + noise)
        augmented.append(df_aug)
    return pd.concat([df] + augmented, ignore_index=True)

# Veri tabanı bağlantısı
engine = create_engine("postgresql://postgres:12345@localhost:5432/database")

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

# Hazırlık ve augmentation
df_prepared = prepare_data(df)
df_augmented = augment_customer_data(df_prepared, n_times=5)

# Özellik ve etiket
features = ['total_orders', 'total_spent', 'avg_order_size'] + [col for col in df_augmented.columns if col.startswith('month_')]
X = df_augmented[features]
y = df_augmented['will_order']

# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sınıf dağılımı
print("Sınıf Dağılımı:\n", y.value_counts())

# SMOTE veya class_weight seçimi
use_smote = False  # ← bunu True yaparsan SMOTE çalışır

if use_smote:
    print("SMOTE uygulanıyor...")
    sm = SMOTE(random_state=42)
    X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
    class_weights = None
else:
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

# Model
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model(X_train_scaled.shape[1])

# Eğitim
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1,
    class_weight=class_weights
)

# Değerlendirme
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Kaydet
model.save("siparis_model_augmented.h5")
joblib.dump(scaler, "scaler_augmented.pkl")
joblib.dump(features, "features_augmented.pkl")
