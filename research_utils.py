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

    # 优先使用 Bing API
    bing_key = os.getenv("BING_API_KEY")
    if bing_key:
        try:
            import urllib.parse
            import urllib.request
            endpoint = "https://api.bing.microsoft.com/v7.0/search"
            params = {
                "q": query,
                "count": num_results,
                "mkt": "zh-CN",
            }
            url = f"{endpoint}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            req.add_header("Ocp-Apim-Subscription-Key", bing_key)
            with urllib.request.urlopen(req, timeout=15) as resp:
                import json as _json
                data = _json.loads(resp.read().decode())
            results = data.get("webPages", {}).get("value", [])
            if not results:
                return "No search results found."
            parts = []
            for i, r in enumerate(results[:num_results], 1):
                title = r.get("name", "")
                snippet = r.get("snippet", "")
                url_link = r.get("url", "")
                parts.append(f"[{i}] {title}\n{snippet}\nURL: {url_link}")
            return _format_results(parts)
        except Exception as e:
            return f"Bing search error: {str(e)}"
    
    # 备选：SerpAPI
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            import urllib.parse
            import urllib.request
            params = {
                "q": query,
                "api_key": serpapi_key,
                "num": num_results,
            }
            url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                import json as _json
                data = _json.loads(resp.read().decode())
            results = data.get("organic_results", [])
            if not results:
                return "No search results found."
            parts = []
            for i, r in enumerate(results[:num_results], 1):
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                parts.append(f"[{i}] {title}\n{snippet}\nURL: {link}")
            return _format_results(parts)
        except Exception as e:
            return f"SerpAPI search error: {str(e)}"
    
    return (
        "Error: No search API key configured. Set BING_API_KEY or SERPAPI_API_KEY in .env. "
        "Bing API: https://azure.microsoft.com/services/cognitive-services/bing-web-search-api/"
    )
