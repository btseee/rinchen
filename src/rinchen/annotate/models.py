# src/rinchen/annotate/models.py
"""Typed row representations for the annotation app's SQLite tables."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Annotator:
    id: int
    name: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Annotator:
        return cls(id=row["id"], name=row["name"], created_at=row["created_at"])


@dataclass(frozen=True, slots=True)
class Item:
    id: int
    set_name: str
    src: str
    src_lang: str
    tgt_lang: str
    doc_id: str
    seg_idx: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Item:
        return cls(
            id=row["id"],
            set_name=row["set_name"],
            src=row["src"],
            src_lang=row["src_lang"],
            tgt_lang=row["tgt_lang"],
            doc_id=row["doc_id"],
            seg_idx=row["seg_idx"],
        )


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    item_id: int
    annotator_id: int | None
    kind: str
    status: str
    created_at: str
    assigned_at: str | None
    completed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Task:
        return cls(
            id=row["id"],
            item_id=row["item_id"],
            annotator_id=row["annotator_id"],
            kind=row["kind"],
            status=row["status"],
            created_at=row["created_at"],
            assigned_at=row["assigned_at"],
            completed_at=row["completed_at"],
        )


@dataclass(frozen=True, slots=True)
class Translation:
    task_id: int
    text: str
    minutes: float
    flags: list[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Translation:
        return cls(
            task_id=row["task_id"],
            text=row["text"],
            minutes=row["minutes"],
            flags=json.loads(row["flags"]),
        )


@dataclass(frozen=True, slots=True)
class Review:
    task_id: int
    score: int
    errors: list[str]
    chosen_translation_id: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Review:
        return cls(
            task_id=row["task_id"],
            score=row["score"],
            errors=json.loads(row["errors"]),
            chosen_translation_id=row["chosen_translation_id"],
        )
