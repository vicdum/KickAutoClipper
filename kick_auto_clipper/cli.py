"""Komut satırı arayüzü.

Örnek:
    kac download https://kick.com/<kanal>/videos/<uuid> --quality 480
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chat import download_chat
from .config import Settings
from .kick_api import KickError, get_vod
from .video import download_video


def vod_dir(workdir: Path, channel: str, uuid: str) -> Path:
    return workdir / channel / uuid


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    vod = get_vod(args.url)
    quality = args.quality or settings.quality
    out = vod_dir(Path(args.workdir or settings.workdir), vod.channel_slug, vod.uuid)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps({**vod.to_dict(), "quality": quality}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"VOD: {vod.title} ({vod.duration / 60:.1f} dk) -> {out}")

    def chat_job() -> str:
        t0 = time.perf_counter()
        n = download_chat(vod, out / "chat.jsonl", workers=settings.chat_workers)
        return f"Sohbet: {n} mesaj ({time.perf_counter() - t0:.1f} sn)"

    def video_job() -> str:
        t0 = time.perf_counter()
        path = download_video(vod, out / "video.mp4", quality=quality)
        return f"Video: {path.name} {quality}p ({time.perf_counter() - t0:.1f} sn)"

    jobs = []
    if not args.video_only:
        jobs.append(chat_job)
    if not args.chat_only:
        jobs.append(video_job)

    # Sohbet ve video indirmeleri birbirinden bağımsız: paralel çalıştır.
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for result in [pool.submit(j) for j in jobs]:
            print(result.result())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kac", description="Kick Auto Clipper")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="VOD videosunu ve sohbet geçmişini indir")
    d.add_argument("url", help="https://kick.com/<kanal>/videos/<uuid>")
    d.add_argument("-q", "--quality", type=int, help="Maksimum yükseklik (ör. 1080, 720, 480). Varsayılan: KAC_QUALITY veya 1080")
    d.add_argument("-o", "--workdir", help="Çıktı klasörü. Varsayılan: KAC_WORKDIR veya workspace")
    g = d.add_mutually_exclusive_group()
    g.add_argument("--chat-only", action="store_true", help="Yalnızca sohbeti indir")
    g.add_argument("--video-only", action="store_true", help="Yalnızca videoyu indir")
    d.set_defaults(func=cmd_download)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, Settings.from_env())
    except KickError as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
