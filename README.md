# tr-news-classification-ann
Türkçe haberleri yapay sinir ağı ile sınıflandıran model.

# Turkish News Classification with ANN (Keras)

Bu proje, Türkçe haberleri 7 farklı kategoriye sınıflandırmak için oluşturulmuş bir yapay sinir ağı (ANN) modelidir.

##  Kategoriler:
- Ekonomi
- Magazin
- Sağlık
- Siyaset
- Spor
- Teknoloji
- Yaşam

##  Kullanılan Teknolojiler:
- Python
- TensorFlow / Keras
- TfidfVectorizer (sklearn)
- LabelEncoder
- Pandas, NumPy

##  Model Özeti:
- 5000 özellikli TF-IDF ile metin vektörleştirme
- Yapay sinir ağı: 128 → 64 → softmax
- %97 test doğruluğu
- Gerçek haberlerle test edildi, %100 tahmin başarısı gösterdi

##  Dosyalar:
- `tr_news_ann_model.keras` → Eğitilmiş model
- `tfidf_vectorizer.pkl` → Vektörleştirici
- `label_encoder.pkl` → Etiket dönüştürücü
- `text_cleaner.py` → Metin ön işleme fonksiyonu
- `predict_test.py` → Örnek tahmin script’i

##  Kullanım:

1. Modeli ve vectorizer’ları yükleyin
2. Haber Metnini Temizleyin ve Vektörleştirin
3. Model ile Tahmin Edin
```python
from tensorflow.keras.models import load_model
import joblib

model = load_model("tr_news_ann_model.keras")
vc = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

text = ["Apple yeni iPhone modelini tanıttı."]
clean = [clean_text(t) for t in text]
vec = vc.transform(clean).toarray()

pred = model.predict(vec).argmax(axis=1)
label = le.inverse_transform(pred)
print("Tahmin:", label[0])
