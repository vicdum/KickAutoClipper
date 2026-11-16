"""Ayarlar: ortam değişkenleri ve isteğe bağlı yerel .env dosyası."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUALITY = 1080


def load_dotenv(path: str | Path = ".env") -> None:
    """Basit .env yükleyici. Zaten tanımlı ortam değişkenlerini ezmez."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    quality: int
    workdir: Path
    chat_workers: int
    encoder: str
    whisper_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            quality=int(os.environ.get("KAC_QUALITY", DEFAULT_QUALITY)),
            workdir=Path(os.environ.get("KAC_WORKDIR", "workspace")),
            chat_workers=int(os.environ.get("KAC_CHAT_WORKERS", 16)),
            encoder=os.environ.get("KAC_ENCODER", "auto"),
            whisper_model=os.environ.get("KAC_WHISPER_MODEL", "large-v3-turbo"),
        )
