# text_processor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

# Отключаем SSL проверку для загрузки NLTK данных (иногда нужно)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Загружает необходимые NLTK-ресурсы с обработкой ошибок."""
    try:
        # Пробуем найти уже скачанные данные
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        # Загружаем данные в пользовательскую папку
        import os
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        
        # Создаем папку, если ее нет
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Загружаем необходимые пакеты
        nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
        nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)
        nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
        
        # Добавляем путь к данным NLTK
        nltk.data.path.append(nltk_data_dir)

def clean_and_normalize(text):
    """Очистка и нормализация английского текста."""
    # Гарантируем, что данные NLTK загружены
    download_nltk_data()
    
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # оставить только буквы и пробелы
    
    # Токенизация
    tokens = word_tokenize(text)
    
    # Загружаем стоп-слова
    stop_words = set(stopwords.words('english'))
    
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    
    # Фильтрация
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    
    return tokens
