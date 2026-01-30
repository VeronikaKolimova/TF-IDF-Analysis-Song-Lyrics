# tfidf.py
import math
from collections import defaultdict


def compute_tf(doc_tokens):
    """Вычисляет TF == Term Frequency для одного документа."""
    tf_dict = defaultdict(float)
    total = len(doc_tokens)
    if total == 0:
        return tf_dict
    for word in doc_tokens:
        tf_dict[word] += 1
    for word in tf_dict:
        tf_dict[word] /= total
    return tf_dict

def compute_idf(corpus_tokens):
    """Вычисляет Inverse Document Frequency для всего корпуса."""
    N = len(corpus_tokens)
    idf_dict = defaultdict(float)
    all_words = set(word for doc in corpus_tokens for word in doc)
    for word in all_words:
        containing_docs = sum(1 for doc in corpus_tokens if word in doc)
        if containing_docs == 0:
            idf_dict[word] = 0.0
        else:
            idf_dict[word] = math.log(N / containing_docs)
    return idf_dict


def compute_tfidf(corpus_tokens):
    """Вычисляет TF-IDF для всего корпуса."""
    idf = compute_idf(corpus_tokens)
    tfidf_corpus = []
    for doc_tokens in corpus_tokens:
        tf = compute_tf(doc_tokens)
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_corpus.append(tfidf)
    return tfidf_corpus


def compute_artist_tfidf(processed_data):
    """Вычисляет TF-IDF для каждого артиста (объединяя все его песни)."""
    # Группируем песни по артистам
    artist_songs = defaultdict(list)
    for item in processed_data:
        artist_songs[item["artist"]].extend(item["tokens"])

    # Вычисляем TF-IDF для каждого артиста
    artist_tfidf = {}
    for artist, tokens in artist_songs.items():
        tf = compute_tf(tokens)
        # Для IDF используем весь корпус (все песни всех артистов)
        corpus_tokens = [item["tokens"] for item in processed_data]
        idf = compute_idf(corpus_tokens)

        artist_tfidf[artist] = {word: tf[word] * idf[word] for word in tf}

    return artist_tfidf


