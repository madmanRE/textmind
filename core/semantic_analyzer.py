import re
import torch
from sentence_transformers import SentenceTransformer, util

# Загружаем модель
MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
ZONES = [
    "title",
    "h1",
    "subheadings",
    "hrefs",
    "first_500_chars",
    "text",
    "url_as_text",
    "structures",
]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text, max_tokens=200):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i: i + max_tokens])


def embed_long_text(text, model, max_tokens=200):
    chunks = list(chunk_text(text, max_tokens))
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings.mean(dim=0)  # усредняем чанки в один вектор


def get_zone_embeddings(docs, zone, model, max_tokens=200):
    """Считает эмбеддинги для конкретной зоны (например 'h1', 'title') с учетом длинных текстов"""
    embeddings = []
    for d in docs:
        value = getattr(d, zone, None)
        if not value:
            continue

        if isinstance(value, list):
            text = " ".join(value)
        else:
            text = str(value)

        text = normalize_text(text)

        if len(text.split()) > max_tokens:
            emb = embed_long_text(text, model, max_tokens)
        else:
            emb = model.encode(text, convert_to_tensor=True)

        embeddings.append(emb)

    if not embeddings:
        return None
    return torch.stack(embeddings)


def compare_zones(my_doc, competitors, zones, model, max_tokens=200):
    results = {}
    for zone in zones:
        # Эмбеддинги конкурентов
        comp_embeds = get_zone_embeddings(competitors, zone, model, max_tokens)
        if comp_embeds is None:
            continue
        comp_mean = comp_embeds.mean(dim=0)

        # Эмбеддинг моего документа
        my_value = getattr(my_doc, zone, 0.0)
        if not my_value:
            continue

        if isinstance(my_value, list):
            my_text = " ".join(my_value)
        else:
            my_text = str(my_value)

        my_text = normalize_text(my_text)

        if len(my_text.split()) > max_tokens:
            my_embed = embed_long_text(my_text, model, max_tokens)
        else:
            my_embed = model.encode(my_text, convert_to_tensor=True)

        # Косинусная близость
        sim = util.cos_sim(my_embed, comp_mean).item()
        results[zone] = sim
    return results


def compute_zone_relevance(MY_DOCUMENT, TOP_COMPETITORS, zones=ZONES, model=MODEL):
    # Считаем релевантность
    zone_relevance = compare_zones(MY_DOCUMENT, TOP_COMPETITORS, zones, model)
    return zone_relevance


def find_semantic_gaps(
    my_doc, competitors, keywords, zones, model, max_tokens=200, top_n=3, min_sim=0.3
):
    # Эмбеддинг ключей
    keywords_embedding = model.encode(" ".join(keywords), convert_to_tensor=True)

    # Эмбеддинг всего текста моего документа
    my_full_text = []
    for zone in zones:
        val = getattr(my_doc, zone, None)
        if val:
            if isinstance(val, list):
                my_full_text.append(" ".join(val))
            else:
                my_full_text.append(str(val))
    my_full_text = normalize_text(" ".join(my_full_text))
    my_full_emb = (
        embed_long_text(my_full_text, model, max_tokens)
        if len(my_full_text.split()) > max_tokens
        else model.encode(my_full_text, convert_to_tensor=True)
    )

    results = {}
    for zone in zones:
        zone_items = []
        for competitor in competitors:
            zone_value = getattr(competitor, zone, None)
            if not zone_value:
                continue
            if isinstance(zone_value, str):
                zone_value = [zone_value]

            for item in zone_value:
                item_text = normalize_text(item)
                item_emb = (
                    embed_long_text(item_text, model, max_tokens)
                    if len(item_text.split()) > max_tokens
                    else model.encode(item_text, convert_to_tensor=True)
                )
                keywords_sim = util.cos_sim(item_emb, keywords_embedding).item()
                if keywords_sim >= min_sim:
                    # Сравнение с зоной моего документа
                    my_zone_value = getattr(my_doc, zone, None)
                    if my_zone_value:
                        if isinstance(my_zone_value, list):
                            my_zone_text = " ".join(my_zone_value)
                        else:
                            my_zone_text = str(my_zone_value)
                        my_zone_text = normalize_text(my_zone_text)
                        my_zone_emb = (
                            embed_long_text(my_zone_text, model, max_tokens)
                            if len(my_zone_text.split()) > max_tokens
                            else model.encode(my_zone_text, convert_to_tensor=True)
                        )
                        my_doc_sim_zone = util.cos_sim(item_emb, my_zone_emb).item()
                        my_zone_kw_sim = util.cos_sim(
                            my_zone_emb, keywords_embedding
                        ).item()
                    else:
                        my_doc_sim_zone = 0.0

                    # Сравнение с полным текстом документа
                    my_doc_sim_full = util.cos_sim(item_emb, my_full_emb).item()

                    zone_items.append(
                        {
                            "competitor": competitor.url,
                            "item": item_text,
                            "keywords_sim": keywords_sim,
                            "my_doc_kw_sim": my_zone_kw_sim,
                            "my_doc_sim_zone": my_doc_sim_zone,
                            "my_doc_sim_full": my_doc_sim_full,
                        }
                    )
        # Берём топ-N элементов по релевантности ключам
        zone_items.sort(key=lambda x: x["keywords_sim"], reverse=True)
        results[zone] = zone_items[:top_n]

    return results


def compute_semantics_gaps(
    MY_DOCUMENT, TOP_COMPETITORS, keyword_list, zones=ZONES, model=MODEL
):
    # Считаем релевантность
    semantic_gaps = find_semantic_gaps(
        MY_DOCUMENT, TOP_COMPETITORS, keyword_list, zones, model
    )
    return semantic_gaps
