"""Komut satırı arayüzü.

    kac run      <vod-url>   Sohbeti indir -> anları tespit et -> sadece klip aralıklarını indir
    kac detect   <vod-url>   İndirilmiş sohbette tespiti yeniden çalıştır (parametre ayarı için)
    kac download <vod-url>   Videonun tamamını ve sohbeti indir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chat import download_chat, load_chat
from .config import Settings
from .detect import Clip, DetectParams, detect
from .hls import MediaPlaylist, download_range, load_playlist
from .kick_api import KickError, Vod, get_vod
from .video import download_video, find_ffmpeg


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Job:
    """Bir VOD için ortak durum: klasör, VOD bilgisi, oynatma listesi."""

    def __init__(self, args: argparse.Namespace, settings: Settings):
        self.settings = settings
        self.quality: int = getattr(args, "quality", None) or settings.quality
        self.vod: Vod = get_vod(args.url)
        self.dir = Path(args.workdir or settings.workdir) / self.vod.channel_slug / self.vod.uuid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.playlist: MediaPlaylist = load_playlist(self.vod.playlist_url, self.quality)
        self.chat_path = self.dir / "chat.jsonl"
        self._write_meta()
        print(f"VOD: {self.vod.title} ({self.vod.duration / 60:.1f} dk, {self.playlist.height}p) -> {self.dir}")

    def _write_meta(self) -> None:
        zero = self.playlist.program_date_time
        meta = {
            **self.vod.to_dict(),
            "quality": self.playlist.height,
            "video_zero": zero.isoformat() if zero else None,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_chat(self, refresh: bool = False) -> list[dict]:
        if refresh or not self.chat_path.exists():
            t0 = time.perf_counter()
            n = download_chat(
                self.vod, self.chat_path,
                workers=self.settings.chat_workers,
                zero=self.playlist.program_date_time,
            )
            print(f"Sohbet: {n} mesaj ({time.perf_counter() - t0:.1f} sn)")
        return load_chat(self.chat_path)


def params_from_args(args: argparse.Namespace) -> DetectParams:
    return DetectParams(
        window=args.window,
        baseline_window=args.baseline,
        ratio=args.ratio,
        min_rate=args.min_rate,
        pad_before=args.pad,
        pad_after=args.pad,
        max_clips=args.max_clips,
    )


def run_detect(job: Job, args: argparse.Namespace) -> list[Clip]:
    chat = job.ensure_chat(getattr(args, "refresh_chat", False))
    clips = detect([m["t"] for m in chat], job.playlist.duration or job.vod.duration, params_from_args(args))
    (job.dir / "clips.json").write_text(
        json.dumps([c.to_dict() for c in clips], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(c.duration for c in clips)
    print(f"Tespit: {len(clips)} an, toplam {total / 60:.1f} dk (videonun %{100 * total / max(job.vod.duration, 1):.1f}'i)")
    for i, c in enumerate(clips, 1):
        print(f"  {i:2d}. {hms(c.start)}-{hms(c.end)}  {c.duration:5.0f} sn  zirve {c.peak_rate:6.1f} msj/dk  x{c.score:.1f}")
    return clips


def cmd_detect(args: argparse.Namespace, settings: Settings) -> int:
    run_detect(Job(args, settings), args)
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise KickError("ffmpeg bulunamadı. Kurup PATH'e ekleyin (Windows: winget install Gyan.FFmpeg).")
    job = Job(args, settings)
    clips = run_detect(job, args)
    if not clips:
        return 0

    clip_dir = job.dir / "clips"
    t0 = time.perf_counter()

    def cut(item: tuple[int, Clip]) -> dict:
        i, c = item
        out = clip_dir / f"clip_{i:02d}_{hms(c.start).replace(':', '')}.mp4"
        actual = download_range(job.playlist, c.start, c.end, out, ffmpeg=ffmpeg)
        # file_start: klip dosyasının ilk karesinin video zamanı (anahtar kareye hizalı)
        return {**c.to_dict(), "file": f"clips/{out.name}", "file_start": round(actual, 3)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(cut, enumerate(clips, 1)))
    (job.dir / "clips.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = [job.dir / r["file"] for r in rows]
    size = sum(p.stat().st_size for p in paths) / 2**20
    print(f"Klipler: {len(paths)} dosya, {size:.0f} MB ({time.perf_counter() - t0:.1f} sn) -> {clip_dir}")
    return 0


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    job = Job(args, settings)

    def video_job() -> str:
        t0 = time.perf_counter()
        path = download_video(job.vod, job.dir / "video.mp4", quality=job.quality)
        return f"Video: {path.name} ({time.perf_counter() - t0:.1f} sn)"

    jobs = []
    if not args.video_only:
        jobs.append(lambda: (job.ensure_chat(refresh=True), "")[1])
    if not args.chat_only:
        jobs.append(video_job)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for msg in [f.result() for f in [pool.submit(j) for j in jobs]]:
            if msg:
                print(msg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kac", description="Kick Auto Clipper")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("url", help="https://kick.com/<kanal>/videos/<uuid>")
        sp.add_argument("-q", "--quality", type=int, help="Maksimum yükseklik (1080, 720, 480...). Varsayılan: KAC_QUALITY veya 1080")
        sp.add_argument("-o", "--workdir", help="Çıktı klasörü. Varsayılan: KAC_WORKDIR veya workspace")

    def detect_opts(sp: argparse.ArgumentParser) -> None:
        d = DetectParams()
        g = sp.add_argument_group("tespit")
        g.add_argument("--window", type=int, default=d.window, help=f"Anlık oran penceresi, sn (varsayılan {d.window})")
        g.add_argument("--baseline", type=int, default=d.baseline_window, help=f"Taban çizgisi penceresi, sn (varsayılan {d.baseline_window})")
        g.add_argument("--ratio", type=float, default=d.ratio, help=f"Eşik = ratio x taban (varsayılan {d.ratio})")
        g.add_argument("--min-rate", type=float, default=d.min_rate, help=f"Minimum mesaj/dk (varsayılan {d.min_rate})")
        g.add_argument("--pad", type=float, default=d.pad_before, help=f"Klip öncesi/sonrası dolgu, sn (varsayılan {d.pad_before})")
        g.add_argument("--max-clips", type=int, default=d.max_clips, help="En yüksek skorlu N klip (0 = hepsi)")
        g.add_argument("--refresh-chat", action="store_true", help="Sohbeti yeniden indir")

    r = sub.add_parser("run", help="Sohbet -> tespit -> yalnızca klip aralıklarını indir")
    common(r)
    detect_opts(r)
    r.set_defaults(func=cmd_run)

    dt = sub.add_parser("detect", help="Sohbetten anları tespit et (video indirmez)")
    common(dt)
    detect_opts(dt)
    dt.set_defaults(func=cmd_detect)

    d = sub.add_parser("download", help="Videonun tamamını ve sohbet geçmişini indir")
    common(d)
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
