"""Модель через прокси организаторов: голый urllib, лимит запросов в минуту, счётчик токенов.

Ключ, адрес и модель берутся только из env (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
Модель рассуждает перед ответом: на это уходит ~500 скрытых токенов, поэтому к max_tokens
добавляется запас, а пустой ответ или обрезка (finish_reason=length) считаются ошибкой.
"""

import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

SSL_CONTEXT = ssl.create_default_context()
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})
PAUSES = (5.0, 10.0, 20.0, 40.0)
THINKING = 2000  # запас на скрытые рассуждения: потолок, а не расход


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reply:
    content: str
    tokens_in: int
    out: int
    finish: str


class Chat(Protocol):
    def ask(self, system: str, user: str, *, max_tokens: int, tag: str) -> Reply: ...


def today() -> str:
    """Сутки квоты считаем по Москве: сервер живёт неделю, лимит обнуляется раз в день."""
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + 3 * 3600))


class Meter:
    """Сколько потрачено за сутки и не больше RPM запросов в минуту на всех потоках."""

    def __init__(self, rpm: int, cap: int) -> None:
        self.rpm = rpm
        self.cap = cap
        self.spent = 0
        self.calls = 0
        self.by_tag: dict[str, int] = {}
        self.day = today()
        self._starts: list[float] = []
        self._lock = threading.Lock()

    def _roll(self) -> None:
        if self.day != today():
            self.day, self.spent, self.calls, self.by_tag = today(), 0, 0, {}

    def wait_slot(self) -> None:
        while True:
            with self._lock:
                self._roll()
                now = time.monotonic()
                self._starts = [t for t in self._starts if now - t < 60]
                if self.cap and self.spent >= self.cap:
                    raise LLMError("исчерпан дневной лимит токенов (TOKEN_CAP)")
                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                sleep = 60 - (now - self._starts[0]) + 0.1
            time.sleep(sleep)

    def add(self, tag: str, tokens: int) -> None:
        with self._lock:
            self._roll()
            self.spent += tokens
            self.calls += 1
            self.by_tag[tag] = self.by_tag.get(tag, 0) + tokens


class Proxy:
    def __init__(self, base_url: str, key: str, model: str, meter: Meter) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = key
        self.model = model
        self.meter = meter
        self.effort = os.environ.get("LLM_REASONING", "low")

    def _post(self, body: bytes, timeout: float) -> tuple[int, str]:
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as r:
                return r.status, r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")

    def ask(self, system: str, user: str, *, max_tokens: int, tag: str) -> Reply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens + THINKING,
            "response_format": {"type": "json_object"},
        }
        if self.effort:
            payload["reasoning_effort"] = self.effort
        body = json.dumps(payload, ensure_ascii=False).encode()
        last = ""
        for attempt in range(len(PAUSES) + 1):
            self.meter.wait_slot()
            try:
                status, text = self._post(body, 120)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                status, text = 0, f"сеть: {type(e).__name__}"
            if status == 200:
                return self._reply(text, tag)
            last = f"HTTP {status}: {text[:160]}"
            if status in (401, 403):
                raise LLMError(f"ключ не принят ({last})")
            if status and status not in RETRY_STATUS:
                raise LLMError(last)
            if attempt < len(PAUSES):
                time.sleep(PAUSES[attempt])
        raise LLMError(f"модель не ответила: {last}")

    def _reply(self, text: str, tag: str) -> Reply:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("ответ прокси не JSON") from e
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        reply = Reply(
            content=str((choice.get("message") or {}).get("content") or ""),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            out=int(usage.get("completion_tokens") or 0),
            finish=str(choice.get("finish_reason") or ""),
        )
        self.meter.add(tag, reply.tokens_in + reply.out)
        if reply.finish == "length":
            raise LLMError("ответ обрезан (finish_reason=length)")
        if not reply.content.strip():
            raise LLMError("пустой ответ модели")
        return reply


def parse_json(text: str) -> dict[str, Any]:
    """JSON из ответа модели, даже если вокруг него текст или ```-обёртка."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("в ответе нет JSON")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError("битый JSON в ответе") from e
    if not isinstance(value, dict):
        raise LLMError("JSON не объект")
    return value


METER = Meter(
    rpm=int(os.environ.get("LLM_RPM", "25")),
    cap=int(os.environ.get("TOKEN_CAP", "200000")),
)


def from_env() -> Chat:
    """Настоящая модель из env, а без ключа или при LLM_FAKE=1 — поддельная."""
    if os.environ.get("LLM_FAKE") == "1" or not os.environ.get("LLM_API_KEY"):
        from sain.fake import FakeChat

        return FakeChat()
    return Proxy(
        os.environ["LLM_BASE_URL"],
        os.environ["LLM_API_KEY"],
        os.environ["LLM_MODEL"],
        METER,
    )
