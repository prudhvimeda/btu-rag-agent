"""Lexical retrieval over module titles: BM25 plus structured filters.

Titles exist in English and German, so a bilingual BM25 index works well without
embeddings and keeps every result explainable (which tokens matched, and how much).
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .catalogue import Catalogue, Module

STOPWORDS = frozenset(
    """a an and are as at be by can course courses credit credits do does ects for from get give
    how in into is it its list me module modules my of offer offered on or semester show tell
    that the their there this to use what when where which who with you your about any are btu
    taught teach teaches take need many much exam exams type kind held run running runs open
    limit limited participant participants place places graded grade grading campus faculty
    cottbus senftenberg year how whether if s class classes lecture lectures student students
    offers provide today tomorrow
    der die das den dem des ein eine einer und oder im mit fur von zu zum zur auf wann ist
    welche welcher welches wie viele wird werden gibt es kann ich modul module kurs""".split()
)
ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "ki": "kunstliche intelligenz",
    "os": "operating systems",
    "db": "databases",
    "dbms": "database systems",
    "it security": "cyber security",
}
MODULE_ID = re.compile(r"\b(\d{5})\b")


def normalize(text: str) -> str:
    text = text.lower().replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _stem(token: str) -> str:
    # Light plural folding shared by both languages: "systems" -> "system", "netze" -> "netz".
    for suffix in ("ies", "es", "s", "en", "e"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix) and not token.endswith("ss"):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    norm = normalize(text)
    for short, long in ALIASES.items():
        norm = re.sub(rf"\b{short}\b", long, norm)
    words = norm.split()
    tokens = []
    for i, word in enumerate(words):
        # "I" is a roman numeral after a title word ("Linear Algebra I"), otherwise a pronoun.
        if word == "i" and (i == 0 or words[i - 1] in STOPWORDS):
            continue
        if word not in STOPWORDS and not word.isdigit():
            tokens.append(_stem(word))
    return tokens


@dataclass(frozen=True)
class Filters:
    level: str | None = None  # "bachelor" | "master"
    language: str | None = None  # "English" | "German"
    semester: str | None = None  # "winter" | "summer"
    programme: str | None = None
    include_inactive: bool = False

    def accepts(self, m: Module) -> bool:
        if not self.include_inactive and not m.active:
            return False
        if self.level and self.level not in m.levels:
            return False
        if self.language and m.language != self.language:
            return False
        if self.semester and m.semester_group not in (self.semester, "every"):
            return False
        if self.programme and self.programme not in m.programmes:
            return False
        return True


@dataclass(frozen=True)
class Hit:
    module: Module
    score: float
    coverage: float  # share of query tokens found in the title
    exact_id: bool = False


@dataclass
class Retriever:
    catalogue: Catalogue
    k1: float = 1.2
    b: float = 0.75
    _docs: list[Counter] = field(init=False, repr=False)
    _df: Counter = field(init=False, repr=False)
    _avg_len: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._docs = [Counter(tokenize(f"{m.title_en} {m.title_de}")) for m in self.catalogue.modules]
        self._df = Counter(t for doc in self._docs for t in doc)
        self._avg_len = sum(sum(d.values()) for d in self._docs) / max(len(self._docs), 1)

    def _idf(self, token: str) -> float:
        n, df = len(self._docs), self._df.get(token, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, filters: Filters = Filters(), k: int = 5) -> list[Hit]:
        exact = [
            Hit(self.catalogue.by_id[i], float("inf"), 1.0, exact_id=True)
            for i in dict.fromkeys(MODULE_ID.findall(query))
            if i in self.catalogue.by_id
        ]
        terms = list(dict.fromkeys(tokenize(query)))
        scored: list[Hit] = []
        if not terms and filters != Filters():
            # A pure filter request ("modules in the AI programme"): list what matches, current ones first.
            scored = [Hit(m, 0.0, 1.0) for m in self.catalogue.modules if filters.accepts(m)]
            scored.sort(key=lambda h: (not h.module.runs_this_semester, h.module.id))
        elif terms:
            for module, doc in zip(self.catalogue.modules, self._docs):
                matched = [t for t in terms if t in doc]
                if not matched or not filters.accepts(module):
                    continue
                length = sum(doc.values())
                score = sum(
                    self._idf(t) * doc[t] * (self.k1 + 1)
                    / (doc[t] + self.k1 * (1 - self.b + self.b * length / self._avg_len))
                    for t in matched
                )
                scored.append(Hit(module, score, len(matched) / len(terms)))
        # Coverage first so a title matching every query word beats one that repeats a rare word.
        scored.sort(key=lambda h: (-round(h.coverage, 3), -h.score, h.module.id))
        seen = {h.module.id for h in exact}
        return (exact + [h for h in scored if h.module.id not in seen])[:k]


def parse_request(text: str, programme_names: tuple[str, ...] = ()) -> tuple[str, Filters]:
    """Rule-based query understanding, used when no model is available and as a baseline."""
    n = normalize(text)
    level = "master" if re.search(r"\b(master|masters|msc|m sc)\b", n) else (
        "bachelor" if re.search(r"\b(bachelor|bachelors|bsc|b sc)\b", n) else None
    )
    semester = "winter" if re.search(r"\b(winter|wise|wintersemester)\b", n) else (
        "summer" if re.search(r"\b(summer|sose|sommersemester)\b", n) else None
    )
    language = "English" if re.search(r"\b(in english|english taught|taught in english|english language)\b", n) else (
        "German" if re.search(r"\b(in german|german taught|taught in german)\b", n) else None
    )
    programme = None
    match = re.search(r"\b(?:in|for) (?:the )?(.+?) (?:programme|program|degree|students)\b", n)
    if match:
        wanted = match.group(1).removesuffix(" master").removesuffix(" bachelor")
        programme = next((p for p in programme_names if normalize(p) == wanted), None)
        if programme:
            n = n.replace(match.group(0), " ")
    n = re.sub(
        r"\b(master|masters|msc|bachelor|bachelors|bsc|winter|summer|wise|sose|english taught|"
        r"taught in english|in english|in german|german taught|taught in german|level|programme|program)\b",
        " ",
        n,
    )
    query = " ".join(MODULE_ID.findall(text) + n.split())
    return query, Filters(level=level, language=language, semester=semester, programme=programme)
