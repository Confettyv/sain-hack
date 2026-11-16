"""HTTP-сервер стенда на stdlib: /run для проверки, /ask и /experts для клиента.

Токен проверки, код набора и данные в лог не попадают: логируем только счётчики.
"""

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sain import llm
from sain.data import DEV_CODE, DataError, Dataset, fetch
from sain.experts import public
from sain.pipeline import answer, ask_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sain")

# Квота модели общая на команду (300k в день), и /run важнее: у клиента свой потолок.
ASK_LIMIT = int(os.environ.get("ASK_LIMIT", "20"))  # вопросов в час
ASK_BUDGET = int(os.environ.get("ASK_BUDGET", "30000"))  # токенов на /ask за жизнь процесса
MAX_BODY = 64 * 1024


class State:
    lock = threading.Lock()
    dataset: Dataset | None = None
    asks: list[float] = []
    ask_tokens = 0


def dataset() -> Dataset:
    """Набор для клиента: последний из /run, иначе dev по токену команды."""
    with State.lock:
        if State.dataset is None:
            token = os.environ.get("HACK_TEAM_TOKEN", "")
            if not token:
                raise DataError("нет HACK_TEAM_TOKEN")
            State.dataset = Dataset.from_payload(fetch(4, DEV_CODE, "Bearer " + token))
        return State.dataset


def ask_allowed() -> bool:
    now = time.time()
    with State.lock:
        State.asks = [t for t in State.asks if now - t < 3600]
        if len(State.asks) >= ASK_LIMIT or State.ask_tokens >= ASK_BUDGET:
            return False
        State.asks.append(now)
        return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — сигнатура базового класса
        pass

    def send_json(self, code: int, body: Any) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            raise ValueError("тело слишком большое")
        data = json.loads(self.rfile.read(size) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("ждём JSON-объект")
        return data

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/", "/health"):
            self.send_json(200, {"ok": True, "team": os.environ.get("TEAM", ""), "case": 4})
        elif path == "/experts":
            try:
                self.send_json(200, {"experts": public(dataset())})
            except DataError as e:
                self.send_json(503, {"error": str(e)})
        elif path == "/stats":
            m = llm.METER
            self.send_json(200, {"tokens": m.spent, "calls": m.calls, "by_tag": m.by_tag, "cap": m.cap, "ask": State.ask_tokens})
        else:
            self.send_json(404, {"error": "нет такого пути"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        try:
            data = self.body()
        except (ValueError, json.JSONDecodeError) as e:
            self.send_json(400, {"error": str(e)})
            return
        if path == "/run":
            self.run_case(data)
        elif path == "/ask":
            self.ask(data)
        else:
            self.send_json(404, {"error": "нет такого пути"})

    def run_case(self, data: dict[str, Any]) -> None:
        auth = self.headers.get("Authorization", "")
        code = str(data.get("dataset_code") or "")
        if not auth.startswith("Bearer ") or not code:
            self.send_json(400, {"error": "нужны Authorization и dataset_code"})
            return
        started = time.monotonic()
        try:
            ds = Dataset.from_payload(fetch(int(data.get("case") or 4), code, auth))
        except DataError as e:
            log.error("run: набор не получен: %s", e)
            self.send_json(502, {"error": str(e)})
            return
        log.info("run: групп %s, запросов %s", len({r.get("group_id") for r in ds.requests}), len(ds.requests))
        result = answer(llm.from_env(), ds)
        with State.lock:
            State.dataset = ds
        m = llm.METER
        log.info("run: готово за %.0f с, вызовов %s, токенов %s %s", time.monotonic() - started, m.calls, m.spent, m.by_tag)
        self.send_json(200, result)

    def ask(self, data: dict[str, Any]) -> None:
        text = str(data.get("text") or "").strip()[:1000]
        expert_id = str(data.get("expert_id") or "")
        history = [
            {"role": str(m.get("role", "user")), "at": "", "text": str(m.get("text", ""))[:400]}
            for m in (data.get("history") or [])[-6:]
            if isinstance(m, dict)
        ]
        try:
            ds = dataset()
        except DataError as e:
            self.send_json(503, {"error": str(e)})
            return
        if not text or expert_id not in ds.experts:
            self.send_json(400, {"error": "нужны text и expert_id из /experts"})
            return
        if not ask_allowed():
            self.send_json(429, {"error": "лимит вопросов на час исчерпан"})
            return
        before = llm.METER.spent
        try:
            self.send_json(200, ask_one(llm.from_env(), ds, expert_id, text, str(data.get("user_id") or ""), history))
        except llm.LLMError as e:
            log.warning("ask: %s", e)
            self.send_json(502, {"error": "модель не ответила"})
        finally:
            with State.lock:
                State.ask_tokens += llm.METER.spent - before


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    log.info("слушаю порт %s, модель %s", port, type(llm.from_env()).__name__)
    server.serve_forever()


if __name__ == "__main__":
    main()
