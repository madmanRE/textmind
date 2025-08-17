from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"


@dataclass
class URLData:
    url: str | None = None
    url_as_text: str | None = None
    title: str | None = None
    h1: str | None = None
    first_500_chars: str | None = None
    subheadings: list[str] | None = None
    hrefs: list[str] | None = None
    structures: list[str] | None = None
    text: str | None = None
    word_count: int | None = None


target_tags = [
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "blockquote",
    "pre",
    "code",
    "div",
    "span",
    "section",
    "article",
    "aside",
    "main",
]


def extract_url_path(url):
    parsed_url = urlparse(url)
    path_text = parsed_url.path
    return path_text.replace("/", " ").replace("-", " ").replace("_", " ")


def build_url_data(url_str, row_html):
    soup = BeautifulSoup(row_html, "html.parser")

    # Удаляем мусор
    for tag in soup(["script", "style", "noscript", "footer", "header", "nav"]):
        tag.extract()

    url_data = URLData()
    url_data.url = url_str

    url_data.url_as_text = extract_url_path(url_str)

    # Title
    title_tag = soup.title
    url_data.title = title_tag.get_text(strip=True) if title_tag else None

    # H1
    h1_tag = soup.find("h1")
    url_data.h1 = h1_tag.get_text(strip=True) if h1_tag else None

    # First 500 chars
    try:
        text_after_h1 = "".join(h1_tag.find_all_next(string=True))
        if text_after_h1:
            first_500_chars = text_after_h1.strip()[:500]
            url_data.first_500_chars = first_500_chars
    except:
        url_data.first_500_chars = None

    # Subheadings
    subheads = []
    for lvl in ["h2", "h3", "h4", "h5", "h6"]:
        for el in soup.find_all(lvl):
            txt = el.get_text(strip=True)
            if txt:
                subheads.append(txt)
    url_data.subheadings = subheads

    # Hrefs
    link_texts = []
    for a in soup.find_all("a", href=True):
        txt = a.get_text(strip=True)
        if txt:
            link_texts.append(txt)

    url_data.hrefs = link_texts if link_texts else None

    # Плоский текст (без списков и таблиц)
    text_parts = []
    for tag in target_tags:
        for element in soup.find_all(tag):
            txt = element.get_text(strip=True)
            if txt:
                text_parts.append(txt)

    full_text = " ".join(text_parts)
    url_data.text = full_text
    url_data.word_count = len(full_text.split())

    # --- СТРУКТУРЫ: списки и таблицы ---
    structures = []

    # Таблицы (каждая строка -> одна строка текста)
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                structures.append(" ".join(cells))

    # Списки (каждый элемент -> отдельная строка)
    for ul in soup.find_all(["ul", "ol"]):
        for li in ul.find_all("li"):
            txt = li.get_text(strip=True)
            if txt:
                structures.append(txt)

    url_data.structures = structures if structures else None

    return url_data


def parse_url(u):
    url_data = URLData(url=u)
    try:
        r = requests.get(
            u,
            headers={
                "Accept-Charset": "utf-8",
                "User-Agent": USER_AGENT,
            },
            timeout=10,
            allow_redirects=True,
        )
        if r.status_code == 200:
            url_data = build_url_data(u, r.text)
        else:
            print(f"Не удалось получить страницу {u}: статус {r.status_code}")
    except requests.RequestException as e:
        print(f"Ошибка при запросе {u}: {e}")
    return url_data


def parse_urls(url_list):
    results = []
    for u in url_list:
        url_data = parse_url(u)
        results.append(url_data)
    return results
