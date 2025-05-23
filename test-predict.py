import pandas as pd

# Test verisini oku ve temizle
test_df = pd.read_csv("test_news.csv")
clean_texts = [clean_text(t) for t in test_df['HABER']]

# Vektörleştir
X_test = vc.transform(clean_texts).toarray()

# Tahmin yap
y_pred = model.predict(X_test).argmax(axis=1)
labels = le.inverse_transform(y_pred)

# Sonuçları göster
test_df['TAHMİN'] = labels
print(test_df)
