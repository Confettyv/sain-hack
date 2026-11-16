"""Оценка V: промпт судьи expert_guess из набора угадывает автора каждого ответа.

  ./.venv/bin/python tools/vcheck.py d4.json answer.json [--limit 12]

V = max(0, (точность − 1/3) / (2/3)). Очистку identity_strip не делаем: роли в наших
ответах не называются, а лишний вызов стоит токенов.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sain import llm  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("answers")
    p.add_argument("--limit", type=int, default=12)
    a = p.parse_args()
    raw = json.loads(Path(a.dataset).read_text())
    raw = raw.get("dataset") or raw
    judge = next(j["text"] for j in raw["judge_prompts"] if j["prompt_id"] == "expert_guess")
    experts = [{k: e.get(k) for k in ("expert_id", "role", "speech_style")} for e in raw["experts"]]
    author = {r["record_id"]: r["expert_id"] for r in raw["requests"]}
    answers = json.loads(Path(a.answers).read_text())["requests"][: a.limit]
    chat = llm.from_env()
    right = 0
    for r in answers:
        user = json.dumps({"experts": experts, "answer": r["text"]}, ensure_ascii=False)
        try:
            guess = llm.parse_json(chat.ask(judge, user, max_tokens=200, tag="judge").content).get("expert_id")
        except (llm.LLMError, ValueError) as e:
            print("судья:", e)
            continue
        ok = guess == author[r["record_id"]]
        right += ok
        if not ok:
            print(f"мимо: {author[r['record_id']]} → {guess}")
    acc = right / (len(answers) or 1)
    print(f"угадано {right}/{len(answers)}, V≈{max(0.0, (acc - 1 / 3) / (2 / 3)):.2f}, токенов {llm.METER.spent}")


if __name__ == "__main__":
    main()
