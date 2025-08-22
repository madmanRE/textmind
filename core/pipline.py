from typing import Any, Dict, List

from core.ai_solver import analyze_results, create_new_page
from core.parser import parse_url, parse_urls
from core.semantic_analyzer import compute_semantics_gaps, compute_zone_relevance


def analyze(
    my_doc: str,
    competitors: List[str],
    keywords: List[str],
    user_agent: str,
    exclude_tags_list: List[str],
    temperatura: int,
    struct: bool,
    new_page: bool = False,
) -> Dict[str, Any]:
    """
    Анализ контента сайта или генерация новой страницы.

    Returns:
        dict:
            {
                "new_page": bool,
                "results": str,  # HTML или текст
                "zone_relevance": dict | None,
                "semantics_gaps": dict | None
            }
    """
    COMPETITORS = parse_urls(competitors, user_agent, exclude_tags_list)

    if new_page:
        results = create_new_page(COMPETITORS, keywords, temperatura)
        return {
            "new_page": True,
            "results": results,
            "zone_relevance": None,
            "semantics_gaps": None,
        }

    MY_DOCUMENT = parse_url(my_doc, user_agent, exclude_tags_list)
    zone_relevance = compute_zone_relevance(MY_DOCUMENT, COMPETITORS)
    semantics_gaps = compute_semantics_gaps(MY_DOCUMENT, COMPETITORS, keywords)
    results = analyze_results(
        MY_DOCUMENT, semantics_gaps, keywords, zone_relevance, temperatura, struct
    )

    return {
        "new_page": False,
        "results": results,
        "zone_relevance": zone_relevance,
        "semantics_gaps": semantics_gaps,
    }
