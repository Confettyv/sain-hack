"""Каталог статей для модели и выбор раздела кодом (BM25 по разделам одной статьи).

Эталон проверяет только статью, поэтому раздел выбирает код: дешевле и без риска
сослаться на несуществующий раздел. Каталог строится из самих данных: новая статья в наборе
сразу попадает в каталог, без шага с моделью.
"""

import math
import re
from collections import Counter

from sain.data import Dataset

SKIP = re.compile(
    # границы слов: «Sleep preferences» и «authority» — не служебные разделы
    r"\b(?:references?\b|acknowledg|funding|conflicts? of|competing|authors?\b|contribution|"
    r"data availab|supplement|abbreviation|disclosure|ethic|declaration|consent|appendix|orcid)",
    re.I,
)
WORD = re.compile(r"[a-z]{3,}")
STOP = frozenset(
    "the and for with that this from were was are been have has not but which their these those "
    "into than also more less such can may our its between during after before over under".split()
)


def words(text: str) -> list[str]:
    out = []
    for w in WORD.findall(text.lower()):
        if w in STOP:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return out


def abstract(ds: Dataset, article_id: str, limit: int = 420) -> str:
    for section in ds.sections.get(article_id, {}).values():
        if "abstract" in section.get("title", "").lower() or section.get("section_id") == "s001":
            text = " ".join(section.get("text", "").split())
            return text[:limit].rsplit(" ", 1)[0] + "…" if len(text) > limit else text
    return ""


def catalog(ds: Dataset) -> str:
    lines = []
    for aid, article in ds.articles.items():
        lines.append(f"{aid} ({article.get('year', '')}): {article.get('title', '')}. {abstract(ds, aid)}")
    return "\n".join(lines)


def usable(ds: Dataset, article_id: str) -> list[dict]:
    sections = [s for s in ds.sections.get(article_id, {}).values() if s.get("text", "").strip()]
    good = [s for s in sections if not SKIP.search(s.get("title", ""))]
    return good or sections


def pick_section(ds: Dataset, article_id: str, query: str) -> str:
    """Раздел статьи, лучше всего совпавший с английскими ключевыми словами; иначе аннотация."""
    sections = usable(ds, article_id)
    if not sections:
        raise KeyError(article_id)
    q = words(query)
    if not q:
        return sections[0]["section_id"]
    docs = [words(s.get("title", "") + " " + s.get("text", "")) for s in sections]
    avg = sum(len(d) for d in docs) / max(1, len(docs))
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    best, best_score = sections[0]["section_id"], 0.0
    for section, doc in zip(sections, docs, strict=True):
        tf = Counter(doc)
        score = 0.0
        for w in set(q):
            if not tf[w]:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            score += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(doc) / avg))
        if score > best_score:
            best, best_score = section["section_id"], score
    return best


def safe_section(ds: Dataset, article_id: str, query: str) -> tuple[str, str]:
    """(статья, раздел) без исключений: у статьи нет разделов — первая статья, где они есть."""
    for a in [article_id, *ds.articles]:
        if a in ds.articles and usable(ds, a):
            return a, pick_section(ds, a, query)
    raise KeyError("в наборе нет статей с разделами")


def excerpt(ds: Dataset, article_id: str, section_id: str, query: str, limit: int = 1600) -> str:
    """Кусок раздела вокруг совпавших слов: модели хватает, а токенов меньше."""
    text = " ".join(ds.sections[article_id][section_id].get("text", "").split())
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    q = set(words(query))
    scored = sorted(
        range(len(sentences)),
        key=lambda i: -len(q & set(words(sentences[i]))),
    )
    keep: set[int] = set()
    size = 0
    for i in scored:
        if size + len(sentences[i]) > limit:
            continue
        keep.add(i)
        size += len(sentences[i]) + 1
    # длинный текст без точек: ни одно предложение не влезло
    return " ".join(sentences[i] for i in sorted(keep)) or text[:limit]
