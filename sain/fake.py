"""Поддельная модель: весь путь работает без токенов (LLM_FAKE=1 или нет ключа).

Тот же контракт, что у настоящей: JSON-строка в Reply.content. Эвристики отсюда же служат
запасным разбором, если модель не ответила на онлайн-прогоне.
"""

import json
import re

from sain.llm import Reply

ATTACK = re.compile(
    r"(сколько\s+(мг|миллиграм|капель|таблет)|доз[аиуы]|мг\b|миллиграм|капел|таблетк|"
    r"что\s+(мне\s+)?купить|в\s+аптеке|похоже\s+на|апноэ|депресси|синдром|диагноз|"
    r"пластыр|розов\w+\s+шум|снотворн)",
    re.I,
)
TOPICS = (
    (r"вин|алкогол|пив|бокал", "art_02", "alcohol heart rate sleep recovery"),
    (r"кофе|кофеин|энергетик|чай", "art_20", "caffeine timing dose sleep"),
    (r"трениров|зал|пробеж|спорт|бег", "art_08", "exercise sleep quality evening"),
    (r"жар|душн|температур|батаре|холод", "art_16", "thermal environment heat awakenings"),
    (r"телефон|экран|сериал|планшет|соцсет", "art_05", "smartphone evening screen sleep"),
    (r"свет|ламп|солнц|утром\s+гул", "art_14", "light exposure evening melatonin"),
    (r"выходн|режим|в\s+разное\s+время|будни", "art_04", "irregular sleep schedule"),
    (r"ужин|ед[аы]|сладк|перекус|молок", "art_06", "diet sleep food"),
    (r"днём\s+посп|дневн\w+\s+сон|вздремн", "art_01", "nap cognitive performance"),
)


def heuristic(text: str, articles: list[str]) -> dict[str, str]:
    kind = "attack" if ATTACK.search(text) else "normal"
    for pattern, article, keywords in TOPICS:
        if re.search(pattern, text, re.I) and article in articles:
            return {"kind": kind, "article": article, "keywords": keywords, "unsafe": "", "topic": ""}
    first = articles[0] if articles else ""
    return {"kind": kind, "article": first, "keywords": "sleep", "unsafe": "", "topic": ""}


FALLBACK = {
    "attack": {
        "clinician": (
            "На этот вопрос я не отвечу заочно: выбор средства, его количество и оценку состояния делает врач "
            "на очном приёме, когда видит анамнез и препараты, которые вы уже принимаете. Из исследований "
            "базы можно взять один безопасный вывод, и он касается привычек вечером, а не лекарств."
        ),
        "biohacker": (
            "Это не то, что тестируют на себе: вещества и дозы — к врачу. Зато можно проверить привычку.\n"
            "Гипотеза: вечерние условия мешают засыпанию.\nМеняем: одну вещь вечером.\n"
            "Меряем: время засыпания в дневнике.\nСрок: неделя.\nКритерий: засыпаешь быстрее."
        ),
        "nutritionist": (
            "Капли, таблетки и добавки пусть подберёт врач, тут я не помощник. А вот за этой просьбой "
            "обычно стоит желание поскорее успокоиться вечером, и это можно поддержать по-бытовому: "
            "одно маленькое изменение после ужина, которое легко повторять каждый день."
        ),
    },
    "normal": {
        "clinician": (
            "Если коротко, исследования базы показывают связь между этой привычкой и качеством сна, "
            "но это связь в среднем по группе, а не гарантия для каждого. Вывод применим к здоровым "
            "взрослым без нарушений сна."
        ),
        "biohacker": (
            "Проверим на тебе.\nГипотеза: привычка влияет на сон.\nМеняем: только её.\n"
            "Меряем: время засыпания и пробуждения в дневнике.\nСрок: две недели.\nКритерий: разница заметна."
        ),
        "nutritionist": (
            "Давай начнём с малого: одна небольшая замена в вечере, привязанная к тому, что ты и так "
            "делаешь, например после ужина. Через неделю посмотри, как спится, и не ругай себя за пропуски."
        ),
    },
}


def fallback(kind: str, expert_id: str) -> str:
    table = FALLBACK["attack" if kind == "attack" else "normal"]
    return table.get(expert_id) or table["clinician"]


class FakeChat:
    def ask(self, system: str, user: str, *, max_tokens: int, tag: str) -> Reply:
        data = json.loads(user)
        if tag == "classify":
            articles = re.findall(r"^(\S+) \(", system, re.M)
            items = [{"n": it["n"], **heuristic(it["text"], articles)} for it in data["items"]]
            content = json.dumps({"items": items}, ensure_ascii=False)
        else:
            content = json.dumps(
                {"answers": {e: fallback(data["kind"], e) for e in data["experts"]}},
                ensure_ascii=False,
            )
        return Reply(content=content, tokens_in=0, out=0, finish="stop")
