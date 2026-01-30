import requests
from bs4 import BeautifulSoup
import json
import time
import os
from requests.exceptions import RequestException, Timeout

# Настройки
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Список артистов (URL-slugs)
artists = [
   "Taylor-swift",
    "Ed-sheeran",
    "Drake",
    "Adele",
    "The-weeknd",
    "Billie-eilish",
    "Kendrick-lamar",
    "Dua-lipa",
    "Post-malone",
    "Rihanna",
    "Bruno-mars",
    "Ariana-grande",
    "Eminem",
    "Coldplay",
    "Beyonce",
    "Harry-styles",
    "Lana-del-rey",
    "Travis-scott",
    "Olivia-rodrigo",
    "Imagine-dragons"
]

output_dir = "lyrics_data"
os.makedirs(output_dir, exist_ok=True)


def safe_get(url, headers, timeout=10):
    try:
        return requests.get(url, headers=headers, timeout=timeout)
    except Timeout:
        print(f" Таймаут: {url}")
        return None
    except RequestException as e:
        print(f" Ошибка сети: {url} — {e}")
        return None

def get_song_links_from_artist(artist_slug, max_songs=15):
    url = f"https://genius.com/artists/{artist_slug}"
    response = safe_get(url, headers)
    if not response:
        return []
    soup = BeautifulSoup(response.text, 'html.parser') # создаем объект для парсинга HTML-страницы с BeautifulSoup
    links = []
    for a in soup.select('a.mini_card'): # CSS-селектор 'a.mini_card' для поиска ссылок на песни на странице артиста
        href = a.get('href')
        if href and href.startswith('https://genius.com/') and 'lyrics' in href:
            links.append(href)
        if len(links) >= max_songs:
            break
    return links

def extract_lyrics_from_page(song_url):
    response = safe_get(song_url, headers)
    if not response:
        return None

    # Принудительно интерпретируем текст как UTF-8, игнорируя недекодируемые байты
    try:
        # response.text уже декодирован, но может содержать "битые" символы
        # Чтобы избежать проблем при сохранении, нормализуем строку
        raw_html = response.content  # получаем байты
        decoded_html = raw_html.decode('utf-8', errors='ignore')  # безопасная декодировка
    except UnicodeDecodeError:
        # fallback: используем response.text, но чистим его
        decoded_html = response.text

    soup = BeautifulSoup(decoded_html, 'html.parser')

    # Способ 1: через JSON (старый формат)
    script_tag = soup.find('script', id='song-json')
    if script_tag and script_tag.string:
        try:
            data = json.loads(script_tag.string)
            lyrics = data.get('song', {}).get('lyrics', '').strip()
            if lyrics:
                # Очистка от недопустимых символов в строке
                lyrics = lyrics.encode('utf-8', errors='ignore').decode('utf-8')
                return lyrics
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            pass

    # Способ 2: через data-lyrics-container (новый формат)
    lyrics_divs = soup.select('div[data-lyrics-container="true"]')
    if lyrics_divs:
        lyrics_parts = []
        for div in lyrics_divs:
            text = div.get_text(separator='\n').strip()
            if text:
                # Очистка каждой части
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
                lyrics_parts.append(text)
        lyrics = '\n'.join(lyrics_parts)
        if lyrics.strip():
            return lyrics

    return None

# Сбор текстов по артистам
all_files = []

for artist in artists:
    print(f"\n Обрабатываем артиста: {artist}")
    artist_lyrics = []
    song_links = get_song_links_from_artist(artist, max_songs=15)

    for i, link in enumerate(song_links):
        print(f"  → Песня {i+1}/{len(song_links)}: {link}")
        lyrics = extract_lyrics_from_page(link)
        if lyrics:
            artist_lyrics.append({
                "artist": artist,
                "song_url": link,
                "lyrics": lyrics
            })
        else:
            print(f"Текст не найден")
        time.sleep(2)  # уважаем сервер

    # Сохраняем по артисту
    if artist_lyrics:
        filename = os.path.join(output_dir, f"lyrics_{artist}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(artist_lyrics, f, ensure_ascii=False, indent=2)
        print(f"Сохранено {len(artist_lyrics)} текстов → {filename}")
        all_files.append(filename)
    else:
        print(f"Нет текстов для {artist}")

# Объединение всех файлов в один общий

print(f"\nОбъединяем все файлы в один общий корпус...")

full_corpus = []
for file in all_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        full_corpus.extend(data)

# Сохраняем общий файл
final_path = os.path.join(output_dir, "lyrics_all.json")
with open(final_path, "w", encoding="utf-8") as f:
    json.dump(full_corpus, f, ensure_ascii=False, indent=2)

print(f"Готово! Всего собрано {len(full_corpus)} текстов.")
print(f"Все файлы сохранены в папке: {output_dir}")
print(f"Общий файд: {final_path}")