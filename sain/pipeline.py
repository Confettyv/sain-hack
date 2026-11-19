"""Путь запроса: тройки → разбор пачкой → раздел кодом → три ответа одним вызовом → проверка.

Ни один сбой не роняет прогон: группа без ответа модели получает запасной текст с той же
ссылкой, потому что запрос без ответа не засчитывается вовсе.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from sain import guard, prompts, search
from sain.data import Dataset
from sain.experts import card
from sain.fake import fallback, heuristic
from sain.llm import Chat, LLMError, parse_json

log = logging.getLogger("sain")
BATCH = 11
WORKERS = 4
DEADLINE = 35 * 60


@dataclass
class Group:
    gid: str
    text: str
    user_id: str
    at: str
    records: list[tuple[str, str]] = field(default_factory=list)  # (record_id, expert_id)
    history: list[dict[str, str]] = field(default_factory=list)  # для клиента: разговор в чате
    last_article: str = ""

    @property
    def experts(self) -> list[str]:
        return list(dict.fromkeys(e for _, e in self.records))


@dataclass
class Decision:
    kind: str
    article: str
    keywords: str
    unsafe: str = ""
    topic: str = ""
    section: str = ""


def groups_of(ds: Dataset) -> list[Group]:
    out: dict[str, Group] = {}
    for r in ds.requests:
        utt = r.get("utterance") or {}
        gid = str(r.get("group_id") or r["record_id"])
        g = out.setdefault(gid, Group(gid, str(utt.get("text", "")), str(r.get("user_id", "")), str(utt.get("at", ""))))
        g.records.append((r["record_id"], r["expert_id"]))
    return list(out.values())


def _history(ds: Dataset, g: Group) -> list[dict[str, str]]:
    return (ds.history(g.user_id, g.at) + g.history)[-6:]


def classify(chat: Chat, ds: Dataset, groups: list[Group], extra: str = "") -> dict[str, Decision]:
    system = prompts.CLASSIFY.replace("{catalog}", search.catalog(ds)) + extra
    articles = list(ds.articles)
    result: dict[str, Decision] = {}
    for start in range(0, len(groups), BATCH):
        chunk = groups[start : start + BATCH]
        items = [
            {"n": i + 1, "text": g.text, "history": [m["text"][:200] for m in _history(ds, g) if m["role"] == "user"][-3:]}
            | ({"last_article": g.last_article} if g.last_article in ds.articles else {})
            for i, g in enumerate(chunk)
        ]
        got: dict[int, dict[str, Any]] = {}
        for attempt in range(2):
            try:
                user = json.dumps({"items": items}, ensure_ascii=False)
                reply = chat.ask(system, user, max_tokens=110 * len(chunk), tag="classify")
                parsed = parse_json(reply.content).get("items", [])
                got = {int(it["n"]): it for it in parsed if isinstance(it, dict) and "n" in it}
                break
            except (LLMError, ValueError, TypeError) as err:
                log.warning("разбор пачки %s, попытка %s: %s", start // BATCH, attempt + 1, err)
        for i, g in enumerate(chunk):
            it = got.get(i + 1) or {}
            guess = heuristic(g.text, articles)
            article = str(it.get("article", ""))
            if article not in ds.articles:
                article = guess["article"]
            kind = it.get("kind") if it.get("kind") in ("attack", "normal") else guess["kind"]
            result[g.gid] = Decision(
                kind=str(kind),
                article=article,
                keywords=str(it.get("keywords") or guess["keywords"]),
                unsafe=str(it.get("unsafe") or ""),
                topic=str(it.get("topic") or ""),
            )
    return result


def write(chat: Chat, ds: Dataset, g: Group, d: Decision, extra: str = "") -> dict[str, str]:
    d.article, d.section = search.safe_section(ds, d.article, d.keywords)
    refused = d.kind == "attack"
    experts = g.experts
    system = prompts.WRITE.replace("{cards}", "\n\n".join(card(ds, e) for e in experts)) + extra
    section = ds.sections[d.article][d.section]
    task: dict[str, Any] = {
        "experts": experts,
        "kind": d.kind,
        "unsafe": d.unsafe,
        "topic": d.topic,
        "utterance": g.text,
        "history": _history(ds, g),
        "user": {k: v for k, v in ds.user(g.user_id).items() if v not in (None, "", [])} or None,
        "article": ds.articles[d.article].get("title", ""),
        "section": section.get("title", ""),
        "excerpt": search.excerpt(ds, d.article, d.section, d.keywords),
    }
    texts: dict[str, str] = {}
    todo = list(experts)
    for attempt in range(2):
        if not todo:
            break
        task["experts"] = todo
        try:
            reply = chat.ask(system, json.dumps(task, ensure_ascii=False), max_tokens=550 * len(todo), tag="write")
            answers = parse_json(reply.content).get("answers") or {}
            if not isinstance(answers, dict):
                raise ValueError("answers не объект")
        except (LLMError, ValueError, TypeError) as err:
            log.warning("ответ группы, попытка %s: %s", attempt + 1, err)
            continue
        bad: list[str] = []
        notes: list[str] = []
        for e in todo:
            text = re.sub(r"[\s{}]+$", "", str(answers.get(e) or "")).strip()
            problems = guard.issues(text, refused)
            if problems:
                bad.append(e)
                notes.append(f"{e}: {', '.join(problems)}")
            else:
                texts[e] = text
        todo = bad
        if bad:
            log.info("фильтр вернул %s", "; ".join(notes))
            task["fix"] = "Прошлый вариант не прошёл проверку: " + "; ".join(notes) + ". Перепиши без этого."
    for e in todo:
        texts[e] = fallback(d.kind, e)
    return texts


def answer(chat: Chat, ds: Dataset, deadline: float = DEADLINE) -> dict[str, Any]:
    started = time.monotonic()
    groups = groups_of(ds)
    decisions = classify(chat, ds, groups)

    def one(g: Group) -> tuple[Group, dict[str, str]]:
        d = decisions[g.gid]
        try:
            if time.monotonic() - started <= deadline:
                return g, write(chat, ds, g, d)
        except Exception:  # noqa: BLE001 — прогон один, группа не должна его ронять
            log.exception("группа упала, запасной ответ")
        if not d.section:
            d.article, d.section = search.safe_section(ds, d.article, d.keywords)
        return g, {e: fallback(d.kind, e) for e in g.experts}

    with ThreadPoolExecutor(WORKERS) as pool:
        done = dict((g.gid, texts) for g, texts in pool.map(one, groups))
    by_record: dict[str, dict[str, Any]] = {}
    for g in groups:
        d = decisions[g.gid]
        for record_id, expert_id in g.records:
            by_record[record_id] = {
                "record_id": record_id,
                "text": done[g.gid][expert_id],
                "refused": d.kind == "attack",
                "citations": [{"article_id": d.article, "section_id": d.section}],
            }
    return {"requests": [by_record[r["record_id"]] for r in ds.requests]}


def ask_live(
    chat: Chat,
    ds: Dataset,
    expert_ids: list[str],
    text: str,
    user_id: str,
    history: list[dict[str, str]],
    last_article: str = "",
) -> dict[str, Any]:
    """Вопрос из клиента: тот же путь, что у /run; несколько экспертов — один вызов на всех."""
    g = Group("live", text, user_id, "", [(e, e) for e in expert_ids], history, last_article)
    d = classify(chat, ds, [g], prompts.LIVE_CLASSIFY)["live"]
    texts = write(chat, ds, g, d, prompts.LIVE_WRITE)
    offtopic = d.topic == "offtopic" and d.kind == "normal"
    article = ds.articles[d.article]
    section = ds.sections[d.article][d.section]
    citation = {
        "article_id": d.article,
        "section_id": d.section,
        "article_title": article.get("title", ""),
        "section_title": section.get("title", ""),
        "year": article.get("year"),
        "doi": article.get("doi", ""),
    }
    return {
        "refused": d.kind == "attack",
        "citations": [] if offtopic else [citation],
        "answers": [{"expert_id": e, "text": texts[e]} for e in expert_ids],
        "degraded": any(texts[e] == fallback(d.kind, e) for e in expert_ids),
    }
