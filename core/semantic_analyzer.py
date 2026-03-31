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


# ------------------ TEXT UTILS ------------------

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text, max_tokens=200):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])


# ------------------ SAFE ENCODE ------------------

def safe_encode(text, model, max_tokens=200, prefix="passage:"):
    text = normalize_text(text)

    if not text:
        return None

    # E5 требует префиксы
    text = f"{prefix} {text}"

    if len(text.split()) > max_tokens:
        chunks = list(chunk_text(text, max_tokens))
        if not chunks:
            return None

        emb = model.encode(chunks, convert_to_tensor=True)

        if emb is None or len(emb) == 0:
            return None

        return emb.mean(dim=0)

    else:
        emb = model.encode(text, convert_to_tensor=True)

        if emb is None:
            return None

        if len(emb.shape) > 1:
            emb = emb.squeeze()

        return emb


# ------------------ ZONE EMBEDDINGS ------------------

def get_zone_embeddings(docs, zone, model, max_tokens=200):
    embeddings = []

    for d in docs:
        value = getattr(d, zone, None)

        if not value or (isinstance(value, list) and not any(value)):
            continue

        text = " ".join(value) if isinstance(value, list) else str(value)

        emb = safe_encode(text, model, max_tokens)

        if emb is not None and emb.shape[0] > 0:
            embeddings.append(emb)

    if not embeddings:
        return None

    # проверка размерности
    dims = [e.shape[0] for e in embeddings]
    if len(set(dims)) != 1:
        return None  # мягкий фейл вместо краша

    return torch.stack(embeddings)


# ------------------ COMPARE ZONES ------------------

def compare_zones(my_doc, competitors, zones, model, max_tokens=200):
    results = {}

    for zone in zones:
        comp_embeds = get_zone_embeddings(competitors, zone, model, max_tokens)
        if comp_embeds is None:
            continue

        comp_mean = comp_embeds.mean(dim=0)

        my_value = getattr(my_doc, zone, None)
        if not my_value:
            continue

        my_text = " ".join(my_value) if isinstance(my_value, list) else str(my_value)

        my_embed = safe_encode(my_text, model, max_tokens)

        if my_embed is None:
            continue

        sim = util.cos_sim(my_embed, comp_mean).item()
        results[zone] = sim

    return results


def compute_zone_relevance(MY_DOCUMENT, TOP_COMPETITORS, zones=ZONES, model=MODEL):
    return compare_zones(MY_DOCUMENT, TOP_COMPETITORS, zones, model)


# ------------------ SEMANTIC GAPS ------------------

def find_semantic_gaps(
    my_doc,
    competitors,
    keywords,
    zones,
    model,
    max_tokens=200,
    top_n=3,
    min_sim=0.3,
):
    # 🔥 keywords как query (важно для e5)
    kw_embeds = [
        safe_encode(k, model, prefix="query:")
        for k in keywords
    ]
    kw_embeds = [k for k in kw_embeds if k is not None]

    if not kw_embeds:
        return {}

    keywords_embedding = torch.stack(kw_embeds).mean(dim=0)

    # 🔥 полный текст документа
    my_full_parts = []

    for zone in zones:
        val = getattr(my_doc, zone, None)
        if val:
            my_full_parts.append(" ".join(val) if isinstance(val, list) else str(val))

    my_full_text = normalize_text(" ".join(my_full_parts))

    my_full_emb = safe_encode(my_full_text, model, max_tokens)
    if my_full_emb is None:
        return {}

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
                item_emb = safe_encode(item, model, max_tokens)

                if item_emb is None:
                    continue

                keywords_sim = util.cos_sim(item_emb, keywords_embedding).item()

                if keywords_sim < min_sim:
                    continue

                # мой zone
                my_zone_value = getattr(my_doc, zone, None)

                if my_zone_value:
                    my_zone_text = (
                        " ".join(my_zone_value)
                        if isinstance(my_zone_value, list)
                        else str(my_zone_value)
                    )

                    my_zone_emb = safe_encode(my_zone_text, model, max_tokens)

                    if my_zone_emb is not None:
                        my_doc_sim_zone = util.cos_sim(item_emb, my_zone_emb).item()
                        my_zone_kw_sim = util.cos_sim(
                            my_zone_emb, keywords_embedding
                        ).item()
                    else:
                        my_doc_sim_zone = 0.0
                        my_zone_kw_sim = 0.0
                else:
                    my_doc_sim_zone = 0.0
                    my_zone_kw_sim = 0.0

                my_doc_sim_full = util.cos_sim(item_emb, my_full_emb).item()

                zone_items.append(
                    {
                        "competitor": competitor.url,
                        "item": item,
                        "keywords_sim": keywords_sim,
                        "my_doc_kw_sim": my_zone_kw_sim,
                        "my_doc_sim_zone": my_doc_sim_zone,
                        "my_doc_sim_full": my_doc_sim_full,
                    }
                )

        zone_items.sort(key=lambda x: x["keywords_sim"], reverse=True)
        results[zone] = zone_items[:top_n]

    return results


def compute_semantics_gaps(
    MY_DOCUMENT, TOP_COMPETITORS, keyword_list, zones=ZONES, model=MODEL
):
    return find_semantic_gaps(
        MY_DOCUMENT, TOP_COMPETITORS, keyword_list, zones, model
    )
