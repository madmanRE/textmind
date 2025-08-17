from typing import List

from core.ai_solver import analyze_results
from core.parser import parse_url, parse_urls
from core.semantic_analyzer import compute_semantics_gaps, compute_zone_relevance


def analyze(my_doc: str, competitors: List[str], keywords: List[str]):
    MY_DOCUMENT = parse_url(my_doc)
    COMPETITORS = parse_urls(competitors)
    zone_relevance = compute_zone_relevance(MY_DOCUMENT, COMPETITORS)
    semantics_gaps = compute_semantics_gaps(MY_DOCUMENT, COMPETITORS, keywords)
    results = analyze_results(MY_DOCUMENT, semantics_gaps, keywords, zone_relevance)
    return zone_relevance, semantics_gaps, results
