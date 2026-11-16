"""Набор кейса 4: загрузка из API хакатона и удобный доступ к статьям, разделам и истории.

URL с кодом набора и токен в лог не пишутся: так требует README стенда.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from sain.llm import SSL_CONTEXT

DEV_CODE = "case4-dev-v2"


class DataError(RuntimeError):
    pass


def fetch(case: int, code: str, authorization: str) -> dict[str, Any]:
    """GET /case/{case}/getData целиком. authorization — заголовок как есть («Bearer …»)."""
    api = os.environ.get("HACK_API_URL", "https://hack-api.be-healthy.tech").rstrip("/")
    url = f"{api}/case/{case}/getData?dataset_code={urllib.parse.quote(code)}"
    request = urllib.request.Request(url, headers={"Authorization": authorization})
    last = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120, context=SSL_CONTEXT) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = type(e).__name__
    raise DataError(f"getData не ответил: {last}")


@dataclass
class Dataset:
    raw: dict[str, Any]
    articles: dict[str, dict[str, Any]] = field(default_factory=dict)
    sections: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    experts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for article in self.raw.get("articles") or []:
            aid = article["article_id"]
            self.articles[aid] = article
            self.sections[aid] = {s["section_id"]: s for s in article.get("sections") or []}
        for expert in self.raw.get("experts") or []:
            self.experts[expert["expert_id"]] = expert

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Dataset":
        """Ответ getData: {"dataset": {...}, "reference": ...} или сразу сам набор."""
        return cls(payload.get("dataset") or payload)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return list(self.raw.get("requests") or [])

    def user(self, user_id: str) -> dict[str, Any]:
        for user in self.raw.get("users") or []:
            if user.get("user_id") == user_id:
                return user
        return {}

    def history(self, user_id: str, before: str = "", limit: int = 6) -> list[dict[str, str]]:
        """Последние сообщения пользователя до реплики: судья видит их и ждёт, что ответ учтёт их."""
        rows = [
            m
            for m in self.raw.get("chat_history") or []
            if m.get("user_id") == user_id and (not before or str(m.get("at", "")) < before)
        ]
        rows.sort(key=lambda m: str(m.get("at", "")))
        return [
            {"role": str(m.get("role", "")), "at": str(m.get("at", ""))[:16], "text": str(m.get("text", ""))[:400]}
            for m in rows[-limit:]
        ]

    def has(self, article_id: str, section_id: str) -> bool:
        return section_id in self.sections.get(article_id, {})
