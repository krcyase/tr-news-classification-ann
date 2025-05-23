import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')




"""
bu kod:
    kelimelere ayırır
    gereksiz kelimeleri temizler.
    sadece harf içerenleri bırakır,
    ve sonucu tekrar metin haline getirir.
    
"""

def clean_text(text):
    text = text.lower() # kelimeleri küçük harflere dönüştürür.
    text = re.sub(r"http\S+|www.\S+", "", text) # linkleri siler.
    text = re.sub(r"<.*?>", "", text) # htlm etiketlerini siler.
    text = re.sub(r"\d+", "", text) #sayıları siler.
    text = text.translate(str.maketrans('', '', string.punctuation)) #noktalama işaretlerini siler.
    tokens = word_tokenize(text, language='turkish') # sözcükleri tokenlara(köklerine) ayırır.
    stop_words = set(stopwords.words('turkish')) # türkçedeki gereksiz kelimeleri silme.
    tokens = [word for word in tokens if word not in  stop_words and word.isalpha()] # iki filtre uygula, gereksiz kelimeleri sil, sadece harflerden oluşanları al.
    return ' '.join(tokens) # filtrelenmiş kelimeleri tekrar tek bir stringe çevir.




