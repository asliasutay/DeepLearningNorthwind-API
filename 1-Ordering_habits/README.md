# Sipariş Verme Alışkanlığı Tahmini

## Veri Mevsimselliği
![image](https://github.com/user-attachments/assets/3fe8df2e-60cb-443a-b963-a76b116005aa)



Bu model, müşterilerin toplam harcaması, sipariş sayısı ve ortalama sipariş büyüklüğüne göre önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin etmektedir.

## Model Özellikleri

- **Girdi Özellikleri:**
  - Toplam sipariş sayısı
  - Toplam harcama miktarı
  - Ortalama sipariş büyüklüğü
  - Aylık sipariş dağılımı (12 ay)

- **Çıktı:**
  - Sipariş verme olasılığı (0-1 arası)
  - Tahmin (Sipariş Verecek/Vermeyecek)

## API Kullanımı

### Endpoint
```
POST /predict
```

### Örnek İstek
```json
{
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
```

### Örnek Yanıt
```json
{
    "siparis_verme_olasiligi": 0.85,
    "tahmin": "Sipariş Verecek"
}
```

## Model Eğitimi

Model eğitimi için `train_augmented.py` ve `sparisverme1.py` dosyaları kullanılmıştır. Eğitim sırasında:

- Veri artırma (Data Augmentation) teknikleri uygulanmıştır
- Mevsimsellik etkisi analiz edilmiştir
- Sınıf dengesizliği (Class Imbalance) sorunu ele alınmıştır

## Dosya Yapısı

- `main.py`: API endpoint'leri
- `sparisverme1.py`: Veri işleme ve model eğitimi
- `train_augmented.py`: Veri artırma ile model eğitimi
- `predict.py`: Tahmin fonksiyonları
- `siparis_model.h5`: Eğitilmiş model
- `scaler.pkl`: Ölçeklendirici
- `features.pkl`: Özellik isimleri 
