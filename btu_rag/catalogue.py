"""Loads the MOFI module catalogue (the same snapshot myBTU ships) into plain records.

The catalogue is stored in a compact form: short keys per module and indices into
lookup tables under ``dict``. ``decode_module`` turns one row into a ``Module``.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

MOFI_ORIGIN = "https://btu-mofi.com"


@dataclass(frozen=True)
class Module:
    id: str
    title_en: str
    title_de: str
    credits: int
    language: str
    faculty: str
    offered: str
    semester_group: str
    exam: str
    graded: bool
    levels: tuple[str, ...]
    programmes: tuple[str, ...]
    campuses: tuple[str, ...]
    participant_limit: int | None
    runs_this_semester: bool
    repeat_exam_only: bool
    attendance_required: bool
    active: bool
    url: str = field(compare=False)

    def evidence(self) -> dict:
        """The fields the model is allowed to state as facts."""
        return {
            "module_id": self.id,
            "title": self.title_en,
            "title_german": self.title_de,
            "credits_ects": self.credits,
            "teaching_language": self.language,
            "faculty": self.faculty,
            "offered": self.offered,
            "exam_type": self.exam,
            "graded": self.graded,
            "levels": list(self.levels),
            "programmes": list(self.programmes[:8]),
            "campuses": list(self.campuses),
            "participant_limit": self.participant_limit,
            "runs_this_semester": self.runs_this_semester,
            "repeat_exam_only": self.repeat_exam_only,
            "attendance_required": self.attendance_required,
            "url": self.url,
        }


@dataclass(frozen=True)
class Catalogue:
    modules: tuple[Module, ...]
    by_id: dict[str, Module]
    programme_names: tuple[str, ...]
    generated_at: str
    schedule_semester: str


def _label(table: list[dict], index: int | None, key: str = "en") -> str:
    if index is None or not 0 <= index < len(table):
        return "Not specified"
    return table[index].get(key) or table[index].get("id") or "Not specified"


def decode_module(row: dict, d: dict) -> Module:
    english_first = row.get("sl") == 1
    primary, secondary = row["t"], row.get("tt") or row["t"]
    programme_by_id = {p["id"]: p["name"] for p in d["programme"]}
    return Module(
        id=str(row["n"]),
        title_en=primary if english_first else secondary,
        title_de=secondary if english_first else primary,
        credits=int(row.get("c") or 0),
        language="English" if row.get("li") == 1 else "German",
        faculty=_label(d["faculty"], row.get("f")),
        offered=_label(d["frequency"], row.get("fq")),
        semester_group=d["frequencyGroup"][row["fg"]]["id"] if row.get("fg") is not None else "unknown",
        exam=_label(d["examType"], row.get("xt")),
        graded=_label(d["grading"], row.get("gr"), "id") == "graded",
        levels=tuple(d["level"][i]["id"] for i in row.get("lv") or [] if i < len(d["level"])),
        programmes=tuple(programme_by_id[p] for p in row.get("pr") or [] if p in programme_by_id),
        campuses=tuple(_label(d["campus"], i) for i in row.get("cp") or []),
        participant_limit=row.get("pl"),
        runs_this_semester=bool(row.get("ev")),
        repeat_exam_only=bool(row.get("rx")),
        attendance_required=bool(row.get("at")),
        active=_label(d["status"], row.get("st_"), "id") == "active",
        url=f"{MOFI_ORIGIN}/modules/{row['n']}/",
    )


def load_catalogue(path: Path | str) -> Catalogue:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw.get("modules"), list) or "dict" not in raw:
        raise ValueError(f"{path} is not a MOFI catalogue")
    modules = tuple(decode_module(row, raw["dict"]) for row in raw["modules"])
    meta = raw.get("meta", {})
    return Catalogue(
        modules=modules,
        by_id={m.id: m for m in modules},
        programme_names=tuple(p["name"] for p in raw["dict"]["programme"]),
        generated_at=meta.get("generatedAt", ""),
        schedule_semester=meta.get("scheduleSemester", ""),
    )


@lru_cache(maxsize=4)
def cached_catalogue(path: str) -> Catalogue:
    return load_catalogue(path)
