"""Эксперты — данные: роль и стиль из набора организаторов + наш рецепт письма из experts/<id>.md.

Новый эксперт = новая запись в наборе; файл с рецептом необязателен, без него модель пишет
по опубликованным роли и стилю.
"""

from pathlib import Path
from typing import Any

from sain.data import Dataset

FOLDER = Path(__file__).resolve().parent.parent / "experts"


def recipe(expert_id: str) -> dict[str, str]:
    path = FOLDER / f"{expert_id}.md"
    if not path.is_file():
        return {"name": expert_id, "tagline": "", "sample": "", "body": ""}
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = {"name": expert_id, "tagline": "", "sample": ""}
    body: list[str] = []
    for line in lines[1:]:
        key, _, value = line.partition(":")
        if not body and key in meta and value:
            meta[key] = value.strip()
        else:
            body.append(line)
    return {**meta, "body": "\n".join(body).strip()}


def card(ds: Dataset, expert_id: str) -> str:
    expert = ds.experts.get(expert_id, {})
    own = recipe(expert_id)
    return (
        f"### {expert_id}\nРоль: {expert.get('role', '')}\nСтиль речи: {expert.get('speech_style', '')}\n{own['body']}"
    ).strip()


def public(ds: Dataset) -> list[dict[str, Any]]:
    """Для клиента: кто есть, чем отличается."""
    out = []
    for expert_id, expert in ds.experts.items():
        own = recipe(expert_id)
        out.append(
            {
                "expert_id": expert_id,
                "name": own["name"],
                "tagline": own["tagline"],
                "sample": own["sample"],
                "role": expert.get("role", ""),
                "speech_style": expert.get("speech_style", ""),
            }
        )
    return out
