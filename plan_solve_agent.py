# -*- coding: utf-8 -*-
"""
Plan-and-Solve Research Agent (with explicit Search Language Selection)

Flow: PLAN (LLM) -> SEARCH (tool-only) -> VERIFY (LLM) -> SOLVE (LLM)

Requirements:
- PLAN must output explicit search languages; must not include web facts.
- SEARCH must only use the language specified by PLAN (no blind translation).
- VERIFY must check constraints + language-context consistency.
- SOLVE must output ONLY the final answer text in the same language as the question.
- Traceability: expose language choice justification, evidence per language, constraint log.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

from research_utils import (
    normalize_answer,
    web_search_raw,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first top-level JSON object from model output.
    """
    if not text:
        raise ValueError("Empty model output")
    # Fast path
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    # Find first '{' and then balance braces
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("Unbalanced JSON braces in model output")


async def _chat_text(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    stream: bool = False,
    timeout_s: int = 600,
) -> str:
    """
    Call DashScope OpenAI-compatible chat API and return full text.
    """
    # Using compatible-mode/v1; timeout handled by upstream HTTP client.
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )
    if not stream:
        return (resp.choices[0].message.content or "").strip()
    # stream=True: accumulate
    out = []
    async for chunk in resp:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            out.append(delta.content)
    return "".join(out).strip()


async def _chat_json(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call model and parse a JSON object from its output.
    """
    text = await _chat_text(client, model, messages, stream=False)
    raw = _extract_first_json_object(text)
    return json.loads(raw)


def _lang_normalize(code: str) -> str:
    code = (code or "").strip().lower()
    if not code:
        return "en"
    # accept zh-cn -> zh, en-us -> en, ja-jp -> ja
    if code.startswith("zh"):
        return "zh"
    if code.startswith("en"):
        return "en"
    if code.startswith("ja") or code.startswith("jp"):
        return "ja"
    if code.startswith("ko"):
        return "ko"
    if code.startswith("fr"):
        return "fr"
    if code.startswith("de"):
        return "de"
    if code.startswith("es"):
        return "es"
    return code


def _question_language_guess(question: str) -> str:
    """
    Very light heuristic. Used for language defaults and output enforcement.
    """
    if re.search(r"[\u4e00-\u9fff]", question or ""):
        return "zh"
    return "en"


def _infer_answer_type(question: str) -> str:
    q = (question or "").lower()
    if re.search(r"\b(year|what year|which year|when)\b", q) or re.search(
        r"(哪一年|年份|年号|何年)", question or ""
    ):
        return "year"
    if re.search(r"\b(how many|number|digit)\b", q) or re.search(r"(多少|几个|数值)", question or ""):
        return "number"
    if re.search(r"\b(who|author|founder|inventor)\b", q) or re.search(
        r"(是谁|作者|创始人|导演|主演)", question or ""
    ):
        return "person_name"
    if re.search(r"\b(company|organization|institution|museum)\b", q) or re.search(
        r"(公司|机构|组织|协会|博物馆)", question or ""
    ):
        return "organization"
    if re.search(r"\b(where|location|city|town|capital)\b", q) or re.search(
        r"(哪里|地点|城市|首都|名称)", question or ""
    ):
        return "place"
    return "text"


def _guess_secondary_languages(question: str) -> List[str]:
    q_lang = _question_language_guess(question)
    if q_lang == "zh":
        return ["en"]
    if re.search(r"\b(china|chinese|taiwan|hong kong)\b", (question or "").lower()):
        return ["zh"]
    return []


def _required_format_hint(question: str) -> str:
    if not question:
        return ""
    if re.search(r"(格式形如|回答格式|要求格式|format)", question, re.IGNORECASE):
        return "Follow the explicit format in the question."
    if re.search(r"(只回答|只输出|answer with|give me)", question, re.IGNORECASE):
        return "Return only the answer text, no explanation."
    return ""


def _is_search_error_result(results: List[Dict[str, Any]]) -> bool:
    if not results:
        return True
    if len(results) == 1 and results[0].get("title") in {
        "BING_ERROR",
        "SERPAPI_ERROR",
        "NO_SEARCH_KEY",
    }:
        return True
    return False


def _filter_valid_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in results or []:
        if r.get("title") in {"BING_ERROR", "SERPAPI_ERROR", "NO_SEARCH_KEY"}:
            continue
        if not r.get("url"):
            continue
        out.append(r)
    return out


@dataclass
class AgentResult:
    answer: str
    trace_id: str
    trace: Dict[str, Any]


class PlanSolveResearchAgent:
    """
    Implements Plan -> Search -> Verify -> Solve with explicit search language selection.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        assert self.api_key, "DASHSCOPE_API_KEY is not set"
        self.model = os.getenv("DASHSCOPE_MODEL", "qwen-max")
        self.client = AsyncOpenAI(
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            base_url = "https://apis.iflow.cn/v1",
            api_key=self.api_key,
        )

    async def run(self, question: str) -> AgentResult:
        trace_id = str(uuid.uuid4())
        trace: Dict[str, Any] = {
            "trace_id": trace_id,
            "created_at": _now_iso(),
            "model": self.model,
            "question": question,
            "phases": {},
        }
        # -----------------
        # 1) LIGHT PLAN (LLM + rules)
        # -----------------
        plan = await self._light_plan(question)
        trace["phases"]["plan"] = plan

        plan_out = plan.get("output", plan)
        primary_lang = _lang_normalize(plan_out.get("primary_language", "en"))
        secondary_langs = [_lang_normalize(x) for x in plan_out.get("secondary_languages", [])]
        allowed_langs = {primary_lang, *secondary_langs} or {"en"}

        # -----------------
        # 2) ITERATIVE SEARCH (tool-only)
        # -----------------
        evidence_by_lang: Dict[str, List[Dict[str, Any]]] = {lang: [] for lang in allowed_langs}
        search_rounds: List[Dict[str, Any]] = []
        seen_queries: Set[str] = set()
        max_rounds = 3
        max_queries_per_lang = 8
        min_valid_results = 4

        queries_by_lang = self._build_initial_queries(question, plan)

        for round_index in range(max_rounds):
            round_runs = []
            total_valid = 0

            for lang, queries in queries_by_lang.items():
                lang = _lang_normalize(lang)
                if lang not in allowed_langs:
                    continue
                lang_runs = []
                for q in queries:
                    q = (q or "").strip()
                    if not q or q in seen_queries:
                        continue
                    if len([x for x in seen_queries if x.startswith(lang + "::")]) >= max_queries_per_lang:
                        continue
                    seen_queries.add(lang + "::" + q)
                    results = web_search_raw(query=q, language=lang, num_results=8)
                    if _is_search_error_result(results):
                        lang_runs.append({"query": q, "language": lang, "results": results, "valid": 0})
                        continue
                    valid = _filter_valid_results(results)
                    total_valid += len(valid)
                    evidence_by_lang.setdefault(lang, []).extend(valid)
                    lang_runs.append({"query": q, "language": lang, "results": results, "valid": len(valid)})
                if lang_runs:
                    round_runs.append({"language": lang, "runs": lang_runs})

            search_rounds.append({"round": round_index + 1, "runs": round_runs})

            if total_valid >= min_valid_results:
                break

            if round_index < max_rounds - 1:
                queries_by_lang = await self._refine_queries(question, plan, evidence_by_lang, allowed_langs)

        trace["phases"]["search"] = {
            "evidence_by_lang": evidence_by_lang,
            "rounds": search_rounds,
        }

        # -----------------
        # 3) VERIFY (LLM, no tools)
        # -----------------
        verify = await self._verify_answer(question, plan, evidence_by_lang)
        trace["phases"]["verify"] = verify

        # -----------------
        # 4) FINAL ANSWER (best_candidate or LLM fallback)
        # -----------------
        best_candidate = (verify.get("best_candidate") or "").strip()
        if best_candidate:
            answer = normalize_answer(best_candidate)
            trace["phases"]["solve"] = {
                "mode": "best_candidate",
                "raw_output": best_candidate,
                "normalized_answer": answer,
            }
        else:
            solved_text = await self._solve_from_evidence(question, plan, evidence_by_lang)
            answer = normalize_answer(solved_text)
            trace["phases"]["solve"] = {
                "mode": "llm_fallback",
                "raw_output": solved_text,
                "normalized_answer": answer,
            }

        trace["answer"] = answer
        trace["finished_at"] = _now_iso()

        # Persist trace for debugging / audit
        try:
            import pathlib

            trace_dir = pathlib.Path(".output") / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{trace_id}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            # Do not fail the request due to trace persistence issues
            pass

        return AgentResult(answer=answer, trace_id=trace_id, trace=trace)

    async def _light_plan(self, question: str) -> Dict[str, Any]:
        q_lang = _question_language_guess(question)
        fallback_plan = {
            "primary_language": q_lang,
            "secondary_languages": _guess_secondary_languages(question),
            "answer_type": _infer_answer_type(question),
            "key_constraints": [],
            "key_terms": [],
            "required_format": _required_format_hint(question),
            "notes": "fallback_plan",
        }

        plan_messages = [
            {
                "role": "system",
                "content": (
                    "You are a lightweight planner for a research agent.\n"
                    "Return ONLY a JSON object, no markdown.\n"
                    "Rules:\n"
                    "- No web access and no guessing answers.\n"
                    "- Identify what the question is asking and extract key constraints.\n"
                    "- Extract key entities, temporal bounds, formats, and answer type.\n\n"
                    "JSON schema:\n"
                    "{\n"
                    '  \"primary_language\": \"<iso code>\",\n'
                    '  \"secondary_languages\": [\"<iso>\"],\n'
                    '  \"answer_type\": \"person_name|organization|number|year|title|device|place|text\",\n'
                    '  \"key_constraints\": [\"...\"],\n'
                    '  \"key_terms\": [\"...\"],\n'
                    '  \"required_format\": \"<format hints from question if any>\"\n'
                    "}\n"
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            plan = await _chat_json(self.client, self.model, plan_messages)
            plan["primary_language"] = _lang_normalize(plan.get("primary_language", q_lang))
            plan["secondary_languages"] = [
                _lang_normalize(x) for x in (plan.get("secondary_languages") or []) if x
            ]
            if not plan.get("answer_type"):
                plan["answer_type"] = fallback_plan["answer_type"]
            if not isinstance(plan.get("key_constraints"), list):
                plan["key_constraints"] = []
            if not isinstance(plan.get("key_terms"), list):
                plan["key_terms"] = []
            if not plan.get("required_format"):
                plan["required_format"] = fallback_plan["required_format"]
            return {"messages": plan_messages, "output": plan}
        except Exception:
            return {"messages": plan_messages, "output": fallback_plan}

    def _build_initial_queries(self, question: str, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        plan_out = plan.get("output", plan)
        key_terms = [t for t in (plan_out.get("key_terms") or []) if t]
        answer_type = plan_out.get("answer_type", "text")
        primary_lang = _lang_normalize(plan_out.get("primary_language", _question_language_guess(question)))
        secondary_langs = [
            _lang_normalize(x) for x in (plan_out.get("secondary_languages") or []) if x
        ]
        languages = [primary_lang] + [x for x in secondary_langs if x != primary_lang]

        query_suffix = {
            "person_name": {"en": "biography", "zh": "人物 简介"},
            "organization": {"en": "official name", "zh": "全称"},
            "year": {"en": "year", "zh": "年份"},
            "number": {"en": "number", "zh": "数字"},
            "title": {"en": "title", "zh": "标题"},
            "place": {"en": "location", "zh": "地点"},
            "device": {"en": "device", "zh": "设备"},
        }

        queries_by_lang: Dict[str, List[str]] = {}
        for lang in languages:
            suffix = query_suffix.get(answer_type, {}).get(lang, "")
            queries = []
            queries.append(question)
            if key_terms:
                queries.append(" ".join(key_terms))
                if suffix:
                    queries.append(" ".join(key_terms + [suffix]))
            elif suffix:
                queries.append(f"{question} {suffix}")
            queries_by_lang[lang] = list(dict.fromkeys([q for q in queries if q.strip()]))

        return queries_by_lang

    async def _refine_queries(
        self,
        question: str,
        plan: Dict[str, Any],
        evidence_by_lang: Dict[str, List[Dict[str, Any]]],
        allowed_langs: Set[str],
    ) -> Dict[str, List[str]]:
        plan_out = plan.get("output", plan)
        seed_terms = plan_out.get("key_terms", [])

        refine_messages = [
            {
                "role": "system",
                "content": (
                    "You create improved search queries.\n"
                    "Return ONLY a JSON object: {\"queries\": {\"<lang>\": [\"q1\", \"q2\"]}}\n"
                    "Rules:\n"
                    "- Use only the specified languages.\n"
                    "- Do not guess the final answer.\n"
                    "- Provide at most 3 queries per language.\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "key_terms": seed_terms,
                        "evidence_snippets": {
                            lang: [r.get("snippet", "") for r in (results or [])[:5]]
                            for lang, results in evidence_by_lang.items()
                        },
                        "languages": sorted(list(allowed_langs)),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            refined = await _chat_json(self.client, self.model, refine_messages)
            queries = refined.get("queries", {})
            out: Dict[str, List[str]] = {}
            for lang in allowed_langs:
                q_list = queries.get(lang, [])
                if isinstance(q_list, list):
                    out[lang] = [q for q in q_list if q and isinstance(q, str)]
            return out
        except Exception:
            return {}

    async def _verify_answer(
        self,
        question: str,
        plan: Dict[str, Any],
        evidence_by_lang: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        plan_out = plan.get("output", plan)
        q_lang = _question_language_guess(question)

        if not any(evidence_by_lang.values()):
            return {
                "messages": [],
                "output": {
                    "selected_languages": {"primary": q_lang, "secondary": []},
                    "language_justification": "no_evidence",
                    "constraint_log": [],
                    "candidates": [],
                    "best_candidate": "",
                    "best_candidate_language": q_lang,
                    "evidence_used": {"by_language": {}},
                },
            }

        verify_messages = [
            {
                "role": "system",
                "content": (
                    "You are the VERIFY module of a research agent.\n"
                    "Return ONLY a JSON object, no markdown.\n"
                    "Choose a best_candidate strictly supported by evidence URLs/snippets.\n"
                    "Do NOT guess. If evidence is weak, set best_candidate to empty string.\n\n"
                    "JSON schema:\n"
                    "{\n"
                    '  \"selected_languages\": {\"primary\": \"<lang>\", \"secondary\": [\"<lang>\"]},\n'
                    '  \"language_justification\": \"<short>\",\n'
                    '  \"constraint_log\": [{\"constraint\": \"...\", \"status\": \"pass|fail|unknown\", \"evidence_urls\": [\"...\"], \"notes\": \"<short>\"}],\n'
                    '  \"candidates\": [{\"value\": \"...\", \"value_language\": \"...\", \"confidence\": 0.0, \"supporting_urls\": [\"...\"], \"supporting_snippets\": [\"...\"], \"violations\": [\"...\"], \"language_consistency\": \"strong|medium|weak\"}],\n'
                    '  \"best_candidate\": \"<value>\",\n'
                    '  \"best_candidate_language\": \"<lang>\",\n'
                    '  \"evidence_used\": {\"by_language\": {\"<lang>\": [{\"url\": \"...\", \"snippet\": \"...\"}]}}\n'
                    "}\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "plan": plan_out,
                        "evidence_by_language": evidence_by_lang,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        verify = await _chat_json(self.client, self.model, verify_messages)
        return {"messages": verify_messages, "output": verify}

    async def _solve_from_evidence(
        self,
        question: str,
        plan: Dict[str, Any],
        evidence_by_lang: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        plan_out = plan.get("output", plan)
        q_lang = _question_language_guess(question)
        top_evidence = {}
        for lang, results in evidence_by_lang.items():
            top_evidence[lang] = [
                {"url": r.get("url", ""), "snippet": r.get("snippet", "")} for r in (results or [])[:8]
            ]

        solve_messages = [
            {
                "role": "system",
                "content": (
                    "You are the SOLVE module. Output ONLY the final answer text and nothing else.\n"
                    "Rules:\n"
                    f"- The answer MUST be in the same language as the original question ({q_lang}).\n"
                    "- Use only the provided evidence snippets and URLs.\n"
                    "- If uncertain, choose the most strongly supported candidate.\n"
                    "- Output must be a single short phrase, no explanation.\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "plan": plan_out,
                        "evidence_by_language": top_evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        return await _chat_text(self.client, self.model, solve_messages, stream=False)

