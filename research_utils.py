# -*- coding: utf-8 -*-
"""
Research Agent 工具：答案归一化、系统提示、联网搜索。
赛题要求：答案简短可核查，使用自建搜索（禁止使用百炼自带搜索）。
"""
import os
import re
from typing import Optional


def normalize_answer(raw: str) -> str:
    """
    赛题答案归一化：小写、去首尾空格；若整段可解析为数字则转为整数字符串。
    """
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    if not s:
        return s
    try:
        n = float(s.replace(",", "").replace(" ", ""))
        if n == int(n):
            return str(int(n))
    except (ValueError, TypeError):
        pass
    return s


RESEARCH_SYSTEM_PROMPT = """You are a Research Agent. Your reply must be ONLY the final answer: exactly one name, one number, or one short phrase. You must NEVER output your reasoning in the final message.

Rules:
1. Always use web_search when the question needs factual or scholarly lookup. Call web_search multiple times with different queries (e.g. different keyword combinations, journal names, years, methodologies) until you find a confident answer. Do not assume the tool is unavailable—always try it.
2. Your final output must be ONLY the answer itself, with no reasoning, no explanation, no background text, no quotes from sources, and no extra sentences. Never output phrases like "the author cannot be determined", "none", "unknown because", or any justification. If truly uncertain after several searches, reply with a single token such as "unknown" (or the best-guess name/number) and nothing else.
3. If the question specifies a format (e.g. zip codes separated by commas), follow it exactly.
4. Use the same language as the question (e.g. Chinese question → Chinese answer) unless the question says otherwise.
5. For numeric questions, give the integer when appropriate (e.g. "140").
6. For company, organization, or person names: use the full official form. Copy the exact form from search results (e.g. "James Lockhart", "RepRap Professional Limited"). Do not abbreviate.
7. For "who is the author" or "which article" questions: run several web_search calls with different queries (e.g. "1972 Latin American Research Review prosopography colonial", "LARR 1972 encomienda author", "article colonial Spanish America prosopography 1972"). Use the snippets to identify the person or title, then output ONLY that name or term."""

# 搜索结果返回给模型时的说明，引导模型照抄原文全称
SEARCH_RESULTS_HEADER = "Use the exact full names and terms as they appear below (do not abbreviate).\n\n"


# 默认返回条数：稍多几条便于模型看到完整名称（Bing 单次最多 50，SerpAPI 一般 10）
DEFAULT_NUM_RESULTS = 8


_LANG_TO_BING_MKT = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
}


def web_search_raw(query: str, language: str = "en", num_results: int = DEFAULT_NUM_RESULTS) -> list:
    """
    返回结构化搜索结果（title/snippet/url），供 Plan-and-Solve 的 SEARCH/VERIFY 使用。
    - SEARCH 阶段必须严格使用 PLAN 指定的 language
    - 本函数不会翻译 query，不会混用语言
    """
    language = (language or "en").strip().lower()
    num_results = min(max(1, int(num_results or DEFAULT_NUM_RESULTS)), 10)

    # Bing API
    bing_key = os.getenv("BING_API_KEY")
    if bing_key:
        try:
            import urllib.parse
            import urllib.request

            endpoint = "https://api.bing.microsoft.com/v7.0/search"
            params = {
                "q": query,
                "count": num_results,
                "mkt": _LANG_TO_BING_MKT.get(language, "en-US"),
            }
            url = f"{endpoint}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            req.add_header("Ocp-Apim-Subscription-Key", bing_key)
            with urllib.request.urlopen(req, timeout=15) as resp:
                import json as _json

                data = _json.loads(resp.read().decode())
            values = data.get("webPages", {}).get("value", []) or []
            out = []
            for r in values[:num_results]:
                out.append(
                    {
                        "title": r.get("name", ""),
                        "snippet": r.get("snippet", ""),
                        "url": r.get("url", ""),
                        "source": "bing",
                        "language": language,
                    }
                )
            return out
        except Exception as e:
            return [
                {
                    "title": "BING_ERROR",
                    "snippet": str(e),
                    "url": "",
                    "source": "bing",
                    "language": language,
                }
            ]

    # SerpAPI
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            import urllib.parse
            import urllib.request

            # SerpAPI 参数参考：hl 控制界面语言；lr 可选语言限制，这里尽量温和不强锁。
            hl = {"zh": "zh-cn", "en": "en", "ja": "ja", "ko": "ko"}.get(language, "en")
            params = {
                "q": query,
                "api_key": serpapi_key,
                "num": num_results,
                "hl": hl,
            }
            url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                import json as _json

                data = _json.loads(resp.read().decode())
            results = data.get("organic_results", []) or []
            out = []
            for r in results[:num_results]:
                out.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "url": r.get("link", ""),
                        "source": "serpapi",
                        "language": language,
                    }
                )
            return out
        except Exception as e:
            return [
                {
                    "title": "SERPAPI_ERROR",
                    "snippet": str(e),
                    "url": "",
                    "source": "serpapi",
                    "language": language,
                }
            ]

    return [
        {
            "title": "NO_SEARCH_KEY",
            "snippet": "No BING_API_KEY or SERPAPI_API_KEY configured.",
            "url": "",
            "source": "none",
            "language": language,
        }
    ]


def web_search(query: str, num_results: int = DEFAULT_NUM_RESULTS) -> str:
    """
    联网搜索。赛题允许：阿里云 IQS、SerpAPI、Bing 等；禁止使用百炼自带的搜索。
    返回结果前会加简短说明，引导模型照抄原文中的全称。
    
    优先级：BING_API_KEY > SERPAPI_API_KEY
    """
    num_results = min(max(1, num_results), 10)  # 1~10 条

    def _format_results(parts: list) -> str:
        body = "\n\n".join(parts)
        return SEARCH_RESULTS_HEADER + body if body else "No search results found."

    # 保持旧工具调用的兼容：默认用中文市场（zh-CN）进行展示型搜索
    raw = web_search_raw(query=query, language="zh", num_results=num_results)
    if raw and raw[0].get("title") in {"BING_ERROR", "SERPAPI_ERROR", "NO_SEARCH_KEY"}:
        return f"Search error: {raw[0].get('snippet', '')}"
    parts = []
    for i, r in enumerate(raw[:num_results], 1):
        parts.append(
            f"[{i}] {r.get('title','')}\n{r.get('snippet','')}\nURL: {r.get('url','')}"
        )
    return _format_results(parts)
