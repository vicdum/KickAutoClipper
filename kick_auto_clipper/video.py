"""VOD videosunu HLS akışından istenen çözünürlükte indirir (yeniden kodlama yok)."""

from __future__ import annotations

import shutil
from pathlib import Path

import yt_dlp

from .kick_api import Vod


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def download_video(vod: Vod, out_path: Path, quality: int = 1080, fragments: int = 16) -> Path:
    """`quality` yüksekliğini aşmayan en iyi akışı indirip MP4 olarak kaydeder."""
    if out_path.exists():
        return out_path
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg bulunamadı. Kurup PATH'e ekleyin (Windows: winget install Gyan.FFmpeg).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        # Kick ana oynatma listesinde video+ses birlikte gelir; yükseklik sınırına göre seç.
        "format": f"best[height<={quality}]/best",
        "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "remux_video": "mp4",
        "concurrent_fragment_downloads": fragments,
        "ffmpeg_location": ffmpeg,
        "noplaylist": True,
        "quiet": True,
        "noprogress": False,
        "retries": 10,
        "fragment_retries": 10,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([vod.playlist_url])
    if not out_path.exists():
        raise RuntimeError(f"Video indirildi ama beklenen dosya oluşmadı: {out_path}")
    return out_path
