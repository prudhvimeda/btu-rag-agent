import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    catalogue_path: Path
    model: str
    judge_model: str
    effort: str
    use_fallbacks: bool
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            catalogue_path=Path(os.getenv("BTU_RAG_CATALOGUE", ROOT / "data" / "mofi-catalog.json")),
            model=os.getenv("BTU_RAG_MODEL", "claude-opus-5-5"),
            judge_model=os.getenv("BTU_RAG_JUDGE_MODEL", "claude-opus-5-5"),
            effort=os.getenv("BTU_RAG_EFFORT", "low"),
            use_fallbacks=os.getenv("BTU_RAG_FALLBACKS", "1") != "0",
            top_k=int(os.getenv("BTU_RAG_TOP_K", "5")),
        )


def has_credentials() -> bool:
    """True when the Anthropic SDK can find a credential without extra setup."""
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
