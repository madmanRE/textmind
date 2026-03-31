import re
import torch
from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer("intfloat/multilingual-e5-large")

ZONES = [
    "title",
    "h1",
    "subheadings",
    "hrefs",
    "first_500_chars",
    "url_as_text",
    "structures",
]

# 🔥 кеш эмбеддингов
EMB_CACHE = {}

# ------------------ TEXT ------------------

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ------------------ ENCODE ------------------

def encode_batch(texts, model, prefix="passage:", batch_size=32):
    texts = [normalize_text(t) for t in texts if t and str(t).strip()]
    if not texts:
        return []

    inputs = [f"{prefix} {t}" for t in texts]

    embeddings = model.encode(
        inputs,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    return embeddings


def cached_encode(text, model, prefix="passage:"):
    text = normalize_text(text)
    if not text:
        return None

    key = f"{prefix}:{text}"

    if key in EMB_CACHE:
        return EMB_CACHE[key]

    emb = model.encode(f"{prefix} {text}", convert_to_tensor=True)

    if emb is not None and len(emb.shape) > 1:
        emb = emb.squeeze()

    EMB_CACHE[key] = emb
    return emb


# ------------------ ZONE EMBEDDINGS ------------------

def get_zone_embeddings(docs, zone, model):
    texts = []

    for d in docs:
        value = getattr(d, zone, None)

        if not value or (isinstance(value, list) and not any(value)):
            continue

        text = " ".join(value) if isinstance(value, list) else str(value)
        text = normalize_text(text)

        if text:
            texts.append(text)

    if not texts:
        return None

    embeddings = encode_batch(texts, model)

    if len(embeddings) == 0:
        return None

    return embeddings


# ------------------ COMPARE ------------------

def compare_zones(my_doc, competitors, zones, model):
    results = {}

    for zone in zones:
        comp_embeds = get_zone_embeddings(competitors, zone, model)
        if comp_embeds is None:
            continue

        comp_mean = comp_embeds.mean(dim=0)

        my_value = getattr(my_doc, zone, None)
        if not my_value:
            continue

        my_text = " ".join(my_value) if isinstance(my_value, list) else str(my_value)

        my_embed = cached_encode(my_text, model)

        if my_embed is None:
            continue

        sim = util.cos_sim(my_embed, comp_mean).item()
        results[zone] = sim

    return results


# ------------------ SEMANTIC GAPS ------------------

def find_semantic_gaps(
    my_doc,
    competitors,
    keywords,
    zones,
    model,
    top_n=3,
    min_sim=0.3,
):
    # 🔥 keywords batch
    kw_embeds = encode_batch(keywords, model, prefix="query:")

    if len(kw_embeds) == 0:
        return {}

    keywords_embedding = kw_embeds.mean(dim=0)

    # 🔥 полный текст
    my_full_text = []

    for zone in zones:
        val = getattr(my_doc, zone, None)
        if val:
            my_full_text.append(" ".join(val) if isinstance(val, list) else str(val))

    my_full_text = normalize_text(" ".join(my_full_text))
    my_full_emb = cached_encode(my_full_text, model)

    if my_full_emb is None:
        return {}

    # 🔥 заранее считаем мои зоны
    my_zone_embeds = {}

    for zone in zones:
        val = getattr(my_doc, zone, None)
        if val:
            text = " ".join(val) if isinstance(val, list) else str(val)
            my_zone_embeds[zone] = cached_encode(text, model)

    results = {}

    for zone in zones:
        texts = []
        meta = []

        for competitor in competitors:
            zone_value = getattr(competitor, zone, None)

            if not zone_value:
                continue

            if isinstance(zone_value, str):
                zone_value = [zone_value]

            for item in zone_value:
                t = normalize_text(item)
                if t:
                    texts.append(t)
                    meta.append((competitor.url, item))

        if not texts:
            continue

        # 🔥 БАТЧ!
        embeddings = encode_batch(texts, model)

        zone_items = []

        for emb, (url, raw_text) in zip(embeddings, meta):
            keywords_sim = util.cos_sim(emb, keywords_embedding).item()

            if keywords_sim < min_sim:
                continue

            my_zone_emb = my_zone_embeds.get(zone)

            if my_zone_emb is not None:
                my_doc_sim_zone = util.cos_sim(emb, my_zone_emb).item()
                my_zone_kw_sim = util.cos_sim(my_zone_emb, keywords_embedding).item()
            else:
                my_doc_sim_zone = 0.0
                my_zone_kw_sim = 0.0

            my_doc_sim_full = util.cos_sim(emb, my_full_emb).item()

            zone_items.append(
                {
                    "competitor": url,
                    "item": raw_text,
                    "keywords_sim": keywords_sim,
                    "my_doc_kw_sim": my_zone_kw_sim,
                    "my_doc_sim_zone": my_doc_sim_zone,
                    "my_doc_sim_full": my_doc_sim_full,
                }
            )

        zone_items.sort(key=lambda x: x["keywords_sim"], reverse=True)
        results[zone] = zone_items[:top_n]

    return results


def compute_zone_relevance(MY_DOCUMENT, TOP_COMPETITORS, zones=ZONES, model=MODEL):
    return compare_zones(MY_DOCUMENT, TOP_COMPETITORS, zones, model)


def compute_semantics_gaps(
    MY_DOCUMENT, TOP_COMPETITORS, keyword_list, zones=ZONES, model=MODEL
):
    return find_semantic_gaps(
        MY_DOCUMENT, TOP_COMPETITORS, keyword_list, zones, model
    )
