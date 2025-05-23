import pandas as pd
# datasetin okunması
df = pd.read_csv('TurkishHeadlines.csv')

from text_cleaner import clean_text
df['clean_haberler'] = df['HABERLER'].apply(clean_text)
print(df.head())

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
le = LabelEncoder()
y = le.fit_transform(df['ETIKET'])
y_cat = to_categorical(y)
print(y[:10])
print(y_cat[:2])

from sklearn.feature_extraction.text import TfidfVectorizer
vc = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vc.fit_transform(df['clean_haberler'])
X_array = X.toarray()
print(X_array.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_array, y_cat, test_size=0.2, stratify=y, random_state=42)

from tensorflow.keras import Sequential
model = Sequential(name='tr_news_ann')

from tensorflow.keras.layers import Input, Dense, Dropout

model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Kayıp Grafiği")
plt.legend(["Eğitim", "Doğrulama"])
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Doğruluk Grafiği")
plt.legend(["Eğitim", "Doğrulama"])
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
y_pred = model.predict(X_test)
y_pred_labels = y_pred.argmax(axis=1)
y_true_labels = y_test.argmax(axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

sns.heatmap(confusion_matrix(y_true_labels, y_pred_labels), annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)

plt.title('Confusion Matrix')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()

eval_result = model.evaluate(X_test, y_test)

for name , value in zip(model.metrics_names, eval_result):
    print(f'{name}:{value}')
    
    
    
test_haberler = [
    "Merkez Bankası faiz oranlarında değişikliğe gitmedi.",
    "Ünlü oyuncu kırmızı halıda verdiği pozlarla gündeme geldi.",
    "Yeni koronavirüs aşısı klinik denemeleri başarıyla tamamladı.",
    "Uzayda gözlem yapan James Webb teleskopu ilginç görüntüler kaydetti.",
    "Fenerbahçe Galatasaray derbisi nefes kesti.",
    "Apple yeni iPhone modelini tanıttı.",
    "Çiftçiler kuraklıktan dolayı büyük zarar gördü."
]

text_clean = [clean_text(text) for text in test_haberler]

test_vec = vc.transform(text_clean).toarray()

pred_probs = model.predict(test_vec)
pred_classes = pred_probs.argmax(axis=1)
predicted_labels = le.inverse_transform(pred_classes)

for i, haber in enumerate(test_haberler):
    print(f"HABER: {haber}")
    print(f"TAHMİN: {predicted_labels[i]}")
    print("-----")

    
model.save('tr_news_ann_model.keras')

import joblib

joblib.dump(vc, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

vc = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")
   
    
    


