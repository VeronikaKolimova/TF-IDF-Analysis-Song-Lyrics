# app.py
import streamlit as st

# ДОЛЖНО БЫТЬ ПЕРВОЙ КОМАНДОЙ STREAMLIT!
st.set_page_config(
    page_title="TF-IDF Анализ Песен",
    page_icon="🎵",
    layout="wide"
)

# Теперь импортируем остальные модули
import json
import os
import sys

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_processor import download_nltk_data, clean_and_normalize
from tfidf import compute_tfidf, compute_artist_tfidf
from collections import Counter
import math

# Инициализация NLTK должна быть первой операцией
try:
    from text_processor import download_nltk_data
    download_nltk_data()
except Exception as e:
    st.error(f"Ошибка инициализации NLTK: {e}")
    st.stop()

# Теперь импортируем остальные модули
from text_processor import clean_and_normalize
from tfidf import compute_tfidf, compute_artist_tfidf
from collections import Counter
import math

# Кэшируем загрузку данных
@st.cache_data(show_spinner="Загрузка данных...")
def load_and_process_data():
    try:
        # Определяем путь к данным (работает локально и на Streamlit Cloud)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "lyrics_data")
        data_path = os.path.join(data_dir, "lyrics_all.json")
        
        # Проверяем существование файла
        if not os.path.exists(data_path):
            # Пробуем альтернативный путь
            data_path = "lyrics_data/lyrics_all.json"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Файл не найден: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Обработка данных
        processed = []
        for item in raw_data:
            lyrics = item.get("lyrics", "")
            if not lyrics or not isinstance(lyrics, str):
                continue
            
            tokens = clean_and_normalize(lyrics)
            if len(tokens) < 10:
                continue
            
            processed.append({
                "artist": item["artist"],
                "song_url": item["song_url"],
                "original_lyrics": lyrics,
                "tokens": tokens
            })
        
        # Вычисляем TF-IDF
        corpus_tokens = [item["tokens"] for item in processed]
        tfidf_scores = compute_tfidf(corpus_tokens)
        artist_tfidf = compute_artist_tfidf(processed)
        
        # Дополнительная статистика
        all_tokens = [token for item in processed for token in item["tokens"]]
        word_freq = Counter(all_tokens)
        total_words = len(all_tokens)
        
        # Вычисляем IDF
        N = len(processed)
        word_idf = {}
        for word in word_freq:
            docs_containing_word = sum(1 for doc_tokens in corpus_tokens if word in doc_tokens)
            word_idf[word] = math.log(N / docs_containing_word) if docs_containing_word > 0 else 0
        
        # Добавляем TF-IDF к каждому документу
        for i, item in enumerate(processed):
            item["tfidf"] = tfidf_scores[i]
        
        return processed, corpus_tokens, tfidf_scores, artist_tfidf, word_freq, total_words, word_idf, N
        
    except Exception as e:
        st.error(f"Ошибка обработки данных: {e}")
        st.write(traceback.format_exc())
        return None, None, None, None, None, None, None, None

def display_top_words(tfidf_dict, title, num_words=10):
    """Утилита для отображения топ-N слов."""
    top_words = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:num_words]

    st.subheader(title)
    if top_words:
        cols = st.columns(2)
        for i, (word, score) in enumerate(top_words):
            with cols[i % 2]:
                st.write(f"- **{word}**: `{score:.4f}`")
    else:
        st.write("Нет значимых слов.")


#  Основной интерфейс Streamlit 
def run_app():
    # ОБНОВЛЕННАЯ КОНФИГУРАЦИЯ СТРАНИЦЫ
    st.set_page_config(
        page_title="TF-IDF Анализ Песен",
        page_icon="🎵",  # фавикон
        layout="wide"
    )

    st.title("TF-IDF Анализ текстов песен")
    st.set_page_config(page_title="TF-IDF Анализ Песен", layout="wide")

    #  Улучшенное объяснение TF-IDF 
    st.markdown("""
    #### TF-IDF (Term Frequency-Inverse Document Frequency) — это статистическая мера, которая показывает, 
    #### насколько важно слово в **конкретном документе** относительно всей коллекции документов (корпуса).
    ####
    #### Формула состоит из двух частей:

    **1. TF (Term Frequency) — Частота термина**  
    > *Насколько часто слово встречается в конкретном документе?*
    """)

    st.latex(
        r"TF(t, d) = \frac{\text{количество вхождений слова t в документ d}}{\text{общее количество слов в документе d}}")

    st.markdown("""
    **2. IDF (Inverse Document Frequency) — Обратная частота документа**  
    > *Насколько редко слово встречается во всех документах коллекции?*
    """)

    st.latex(
        r"IDF(t, D) = \log\left(\frac{\text{общее количество документов в корпусе D}}{\text{количество документов, содержащих слово t}}\right)")

    st.markdown("""
    **3. TF-IDF — итоговая мера**  
    > *Комбинация частоты и уникальности слова:*
    """)

    st.latex(r"TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)")

    st.markdown("""
    - **Высокий TF-IDF** = слово часто встречается в этом документе, но редко в других → **характерное, уникальное слово**
    - **Низкий TF-IDF** = слово редко в документе или часто в других → **обычное, распространенное слово**

    В нашем случае:  
    - **Документ** = текст песни (или все песни артиста)  
    - **Корпус** = 200 песен 20 артистов

    ***Этот анализ выполняется с помощью ручной реализации алгоритма TF-IDF.***
    """)

    st.markdown("---")

    try:
        processed_data, _, _, artist_tfidf, word_freq, total_words, word_idf, N = load_and_process_data()
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return

    if not processed_data:
        st.warning("Нет данных для отображения. Проверьте файл lyrics_all.json.")
        return

    #  Статистика по корпусу
    st.header("Общая статистика по корпусу")
    st.markdown(
        f"Всего **{len(processed_data)} песен** от **{len(set(item['artist'] for item in processed_data))} артистов**")
    st.markdown(f"Всего **{total_words} слов** (**{len(word_freq)} уникальных слов**)")

    # Самые частые слова в корпусе
    top_frequent = dict(word_freq.most_common(15))
    st.subheader("Самые частые слова во всем корпусе")
    for word, freq in list(top_frequent.items())[:10]:
        st.write(f"- **{word}**: встречается {freq} раз ({(freq / total_words * 100):.2f}%)")

    # Слова с highest IDF (самые редкие)
    top_idf = sorted(word_idf.items(), key=lambda x: x[1], reverse=True)[:10]
    st.subheader("Самые редкие слова в корпусе (highest IDF)")
    st.markdown("встречаются в наименьшем количестве песен:")
    for word, idf_score in top_idf:
        docs_with_word = sum(1 for doc_tokens in [item["tokens"] for item in processed_data] if word in doc_tokens)
        st.write(f"- **{word}**: IDF = `{idf_score:.4f}` (всего в {docs_with_word} песнях из {N})")

    st.markdown("---")

    #  TF-IDF по артистам 
    st.header("TF-IDF по артистам")
    st.markdown("""
    **Рассматриваем все песни артиста как один "документ" 
    и сравниваем с корпусом всех песен.
    """)

    artists = sorted(set(item["artist"] for item in processed_data))
    selected_artist_stats = st.selectbox("Выберите исполнителя для анализа", artists, key="artist_stats")

    if selected_artist_stats in artist_tfidf:
        display_top_words(artist_tfidf[selected_artist_stats],
                          f"Топ-15 характерных слов для {selected_artist_stats}", 15)

        # Пояснение для артиста
        st.info(f"""
        **Интерпретация для {selected_artist_stats}:**
        - Эти слова часто встречаются в песнях этого артиста (высокий TF)
        - Но при этом они редко встречаются у других артистов (высокий IDF)
        - Это делает их **уникальными** для творчества {selected_artist_stats}
        """)
    else:
        st.warning(f"Нет данных для артиста {selected_artist_stats}")

    st.markdown("---")

    #  TF-IDF по отдельным песням 
    st.header("TF-IDF по отдельным песням")
    st.markdown("""
    **Классическое применение TF-IDF:** анализ конкретной песни относительно всей коллекции.
    """)

    selected_artist = st.selectbox("Выберите исполнителя", artists, key="artist_songs")

    artist_songs = []
    for i, item in enumerate(processed_data):
        if item["artist"] == selected_artist:
            slug = item["song_url"].split("/")[-1]
            if slug.endswith("-lyrics"):
                slug = slug[:-7]
            name = slug.replace("-", " ").title()
            artist_songs.append((i, name))

    song_names = [name for _, name in artist_songs]
    if song_names:
        selected_song_index = st.selectbox("Выберите песню", range(len(song_names)),
                                           format_func=lambda x: song_names[x])

        doc_index = artist_songs[selected_song_index][0]
        tfidf_dict = processed_data[doc_index]["tfidf"]

        display_top_words(tfidf_dict, "Топ-10 слов с наибольшим TF-IDF в этой песне:", 10)

        # Пояснение для песни
        st.info(f"""
        **Интерпретация для песни "{song_names[selected_song_index]}":**
        - Эти слова являются ключевыми для данной конкретной песни
        - Они часто встречаются в этой песне (высокий TF), но редко в других песнях коллекции (высокий IDF)
        - Это **самые характерные слова** именно для этой композиции
        """)

        with st.expander("Показать оригинальный текст песни"):
            original_text = processed_data[doc_index]["original_lyrics"]
            if original_text.strip():
                st.text_area("Текст песни", original_text, height=300, disabled=True, key="lyrics_text")
            else:
                st.write("Оригинальный текст отсутствует.")
    else:
        st.warning(f"Нет песен для артиста {selected_artist}")


run_app()
