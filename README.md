# Northwind Veritabanı Derin Öğrenme Projeleri

Bu proje, Northwind veritabanı üzerinde üç farklı derin öğrenme modeli içermektedir:

## 1. Sipariş Verme Alışkanlığı Tahmini
Müşterilerin toplam harcaması, sipariş sayısı ve ortalama sipariş büyüklüğüne göre önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin eden model.

### API Endpoint
- POST `/predict`
- Girdi: Müşteri sipariş özellikleri
- Çıktı: Sipariş verme olasılığı ve tahmin

## 2. Ürün İade Riski Skoru
Müşterilerin daha önceki siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre bir siparişin iade edilme riskini tahmin eden model.

### API Endpoint
- POST `/predict`
- Girdi: Ürün sipariş detayları
- Çıktı: İade riski olasılığı ve tahmin

## 3. Yeni Ürün Satın Alma Potansiyeli
Müşterilerin geçmiş satın alma kategorilerine bakarak, yeni çıkan bir ürünü satın alma ihtimallerini tahmin eden model.

### API Endpoint
- POST `/predict`
- Girdi: Müşteri kategori harcamaları
- Çıktı: Her kategori için satın alma olasılıkları

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Her bir model için ayrı ayrı API'yi çalıştırın:
```bash
# Sipariş Verme Alışkanlığı API
uvicorn first.main:app --reload

# Ürün İade Riski API
uvicorn second.main:app --reload

## 👨‍💻 Katkı Sağlayanlar

- Aslı Asutay
- Eslem Nur Gök
- Gül Erten
- Yağmur Polat
- Nour Baroudi


# Yeni Ürün Satın Alma Potansiyeli API
uvicorn third.main:app --reload
```

## Model Detayları

Her bir model için detaylı bilgi ve kullanım örnekleri ilgili klasörlerdeki README dosyalarında bulunmaktadır. 
