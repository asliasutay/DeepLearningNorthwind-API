# Ürün İade Riski Skoru

Bu model, müşterilerin daha önceki siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre bir siparişin iade edilme riskini tahmin etmektedir.

## Model Özellikleri

- **Girdi Özellikleri:**
  - İndirim oranı
  - Ürün miktarı
  - Birim fiyat

- **Çıktı:**
  - İade riski olasılığı (0-1 arası)
  - Tahmin (İade riski yüksek/düşük)

## API Kullanımı

### Endpoint
```
POST /predict
```

### Örnek İstek
```json
{
    "discount": 0.15,
    "quantity": 5,
    "unit_price": 25.0
}
```

### Örnek Yanıt
```json
{
    "risk_olasiligi": 0.75,
    "tahmin": "İade riski yüksek"
}
```

## Model Eğitimi

Model eğitimi için `productrisk.py` dosyası kullanılmıştır. Eğitim sırasında:

- Maliyet duyarlı öğrenme (Cost-sensitive Learning) uygulanmıştır
- İade riski yüksek siparişler için özel ağırlıklandırma yapılmıştır
- SHAP ve LIME gibi açıklanabilir AI teknikleri kullanılmıştır

## Dosya Yapısı

- `main.py`: API endpoint'leri
- `productrisk.py`: Veri işleme ve model eğitimi
- `product_model.h5`: Eğitilmiş model
- `product_scaler.pkl`: Ölçeklendirici 