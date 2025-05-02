# Yeni Ürün Satın Alma Potansiyeli

Bu model, müşterilerin geçmiş satın alma kategorilerine bakarak, yeni çıkan bir ürünü satın alma ihtimallerini tahmin etmektedir.

## Model Özellikleri

- **Girdi Özellikleri:**
  - Beverages (İçecekler)
  - Condiments (Baharatlar)
  - Confections (Şekerlemeler)
  - Dairy Products (Süt Ürünleri)
  - Grains/Cereals (Tahıllar)
  - Meat/Poultry (Et/Kümes Hayvanları)
  - Produce (Sebze/Meyve)
  - Seafood (Deniz Ürünleri)

- **Çıktı:**
  - Her kategori için satın alma olasılığı (0-1 arası)

## API Kullanımı

### Endpoint
```
POST /predict
```

### Örnek İstek
```json
{
    "Beverages": 1000.0,
    "Condiments": 500.0,
    "Confections": 800.0,
    "Dairy_Products": 600.0,
    "Grains_Cereals": 400.0,
    "Meat_Poultry": 700.0,
    "Produce": 300.0,
    "Seafood": 900.0
}
```

### Örnek Yanıt
```json
{
    "tahmin_edilen_kategori_olasiliklari": {
        "Beverages": 0.85,
        "Condiments": 0.45,
        "Confections": 0.75,
        "Dairy Products": 0.65,
        "Grains/Cereals": 0.35,
        "Meat/Poultry": 0.55,
        "Produce": 0.25,
        "Seafood": 0.95
    }
}
```

## Model Eğitimi

Model eğitimi için `order_prediction.py` dosyası kullanılmıştır. Eğitim sırasında:

- Neural Collaborative Filtering teknikleri uygulanmıştır
- AutoEncoder tabanlı öneri sistemi kullanılmıştır
- Çoklu etiket tahmini (Multi-label Prediction) yapılmıştır

## Dosya Yapısı

- `main.py`: API endpoint'leri
- `order_prediction.py`: Veri işleme ve model eğitimi
- `predict.py`: Tahmin fonksiyonları
- `category_multi_model.h5`: Eğitilmiş model
- `scaler_multi.pkl`: Ölçeklendirici
- `category_names.pkl`: Kategori isimleri 