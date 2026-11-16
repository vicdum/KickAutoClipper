"""Komut satırı arayüzü.

    kac run      <vod-url>   Sohbeti indir -> anları tespit et -> sadece klip aralıklarını indir
    kac detect   <vod-url>   İndirilmiş sohbette tespiti yeniden çalıştır (parametre ayarı için)
    kac download <vod-url>   Videonun tamamını ve sohbeti indir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chat import download_chat, load_chat
from .config import Settings
from .detect import Clip, DetectParams, detect
from .hls import MediaPlaylist, cut, fetch_range, load_playlist
from .kick_api import KickError, Vod, get_vod
from .render import ENCODERS, LAYOUTS, OutputFormat, resolve_encoder
from . import speech
from .subtitles import Transcriber, load_stereo, shift, to_ass, to_srt, write_wav

CONTEXT = 15.0  # altyazı tanımada klipten önce modele verilen bağlam (sn)
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
        intro=args.intro,
        outro=args.outro,
        max_duration=args.max_duration,
        max_clips=args.max_clips,
    )


def run_detect(job: Job, args: argparse.Namespace) -> list[Clip]:
    chat = job.ensure_chat(getattr(args, "refresh_chat", False))
    result = detect(
        chat, job.playlist.duration or job.vod.duration, params_from_args(args),
        interruptions=job.playlist.interruptions,
    )
    clips = result.clips
    (job.dir / "clips.json").write_text(
        json.dumps([c.to_dict() for c in clips], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(c.duration for c in clips)
    print(f"Tespit: {len(clips)} an, toplam {total / 60:.1f} dk (videonun %{100 * total / max(job.vod.duration, 1):.1f}'i)")
    for i, c in enumerate(clips, 1):
        print(f"  {i:2d}. {hms(c.start)}-{hms(c.end)}  {c.duration:3.0f} sn  zirve {c.peak_rate:6.1f} msj/dk  x{c.score:.1f}")
    if result.rejected:
        print(f"Elenen: {len(result.rejected)} sıçrama")
        for r in result.rejected:
            c = r.clip
            noise = ", ".join(f"{k} %{v * 100:.0f}" for k, v in c.noise.items())
            print(f"   - {hms(c.spike_start)}-{hms(c.spike_end)}  x{c.score:.1f}  {r.reason}" + (f"  ({noise})" if noise else ""))
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

    encoder = resolve_encoder(args.encoder or settings.encoder, ffmpeg)
    fmt = OutputFormat.for_quality(job.playlist.height, args.layout, encoder)
    clip_dir = job.dir / "clips"
    for old in clip_dir.glob("clip_*.*"):  # önceki çalıştırmanın çıktıları
        old.unlink()

    transcriber = None
    if args.subtitles:
        t0 = time.perf_counter()
        model = args.whisper_model or settings.whisper_model
        transcriber = Transcriber(model)
        print(f"Altyazı modeli: {model} ({transcriber.device}, {time.perf_counter() - t0:.1f} sn)")
    boost = args.voice_boost and fmt.layout != "original"
    if boost and not speech.available():
        print("Konuşma güçlendirme için PyTorch gerekli (pip install -e \".[gpu]\"); atlanıyor.")
        boost = False
    t0 = time.perf_counter()

    def make_clip(item: tuple[int, Clip]) -> dict:
        i, c = item
        name = f"clip_{i:02d}_{hms(c.start).replace(':', '')}"
        out = clip_dir / f"{name}.mp4"
        t = [time.perf_counter()]
        # Altyazı bağlamı için klipten önceki CONTEXT saniye de indirilir.
        src = fetch_range(job.playlist, max(0.0, c.start - (CONTEXT if transcriber else 0)), c.end, clip_dir)
        t.append(time.perf_counter())
        ass, voice, cues = None, None, []
        try:
            if transcriber:
                # Ses, kesilecek aralığın aynısından okunur; zamanlar c.start'a göredir.
                cues = transcriber.cues(src.path, job.vod.language, src.offset(c.start), c.duration,
                                       context=CONTEXT, refine=args.refine_timing,
                                       multilingual=not args.single_language)
                if fmt.layout != "original" and cues:
                    ass = f"{name}.ass"
                    (clip_dir / ass).write_text(to_ass(cues, fmt.width, fmt.height, language=job.vod.language), encoding="utf-8")
            t.append(time.perf_counter())
            if boost:
                voice = clip_dir / f"{name}.voice.wav"
                write_wav(voice, speech.voice_boost(load_stereo(src.path, src.offset(c.start), c.duration), 44100), 44100)
            t.append(time.perf_counter())
            encode_args = None if fmt.layout == "original" else fmt.ffmpeg_args(subtitles=ass, external_audio=voice is not None)
            actual = cut(src, c.start, c.end, out, ffmpeg=ffmpeg, encode_args=encode_args,
                         extra_inputs=[voice] if voice else None)
            t.append(time.perf_counter())
            parts = [f"indirme {t[1] - t[0]:4.1f} sn"]
            if transcriber:
                parts.append(f"altyazı {t[2] - t[1]:4.1f} sn ({len(cues)} satır)")
            if boost:
                parts.append(f"ses {t[3] - t[2]:4.1f} sn")
            parts.append(f"kodlama {t[4] - t[3]:4.1f} sn")
            print(f"  {name}: " + ", ".join(parts))
            if transcriber:
                # Kodlamasız kesim anahtar kareye çekildiyse altyazıyı aynı miktar kaydır.
                (clip_dir / f"{name}.srt").write_text(to_srt(shift(cues, c.start - actual)), encoding="utf-8")
        finally:
            src.path.unlink(missing_ok=True)
            for tmp in (ass and clip_dir / ass, voice):  # videoya gömüldü; SRT ayrıca saklanır
                if tmp:
                    tmp.unlink(missing_ok=True)
        # file_start: klip dosyasının ilk karesinin video zamanı
        return {**c.to_dict(), "file": f"clips/{out.name}", "file_start": round(actual, 3)}

    layout = "yatay, kodlamasız" if fmt.layout == "original" else f"{fmt.width}x{fmt.height} {fmt.layout}, {fmt.encoder}"
    print(f"Klipler hazırlanıyor ({layout}{', altyazılı' if transcriber else ''})...")
    # libx264 zaten tüm çekirdekleri kullanır; GPU kodlamada filtreler CPU'da kaldığı için 3 iş iyi dengeler.
    # Altyazı tanıma GPU'da sıraya alınır; bu sırada diğer klipler indirilip kodlanır.
    workers = int(os.environ.get("KAC_CLIP_WORKERS", 2 if fmt.encoder == "cpu" and fmt.layout != "original" else 3))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(make_clip, enumerate(clips, 1)))
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
        g.add_argument("--intro", type=float, default=d.intro, help=f"Yayının ilk N sn'sindeki sıçramaları yok say (varsayılan {d.intro:.0f})")
        g.add_argument("--outro", type=float, default=d.outro, help=f"Yayının son N sn'sindeki sıçramaları yok say (varsayılan {d.outro:.0f})")
        g.add_argument("--max-duration", type=float, default=d.max_duration, help=f"Klip üst sınırı, sn (varsayılan {d.max_duration:.0f}; Shorts/Reels 180, TikTok 600'e kadar izin verir)")
        g.add_argument("--max-clips", type=int, default=d.max_clips, help="En yüksek skorlu N klip (0 = hepsi)")
        g.add_argument("--refresh-chat", action="store_true", help="Sohbeti yeniden indir")

    r = sub.add_parser("run", help="Sohbet -> tespit -> yalnızca klip aralıklarını indir")
    common(r)
    detect_opts(r)
    r.add_argument("--layout", choices=LAYOUTS, default="blur",
                   help="Dikey yerleşim: blur (bulanık arka plan, varsayılan), crop (ortadan kesit), original (yatay, kodlamasız)")
    r.add_argument("--encoder", choices=ENCODERS,
                   help="Video kodlayıcı: auto (NVIDIA GPU varsa onu kullanır), nvenc, cpu. Varsayılan: KAC_ENCODER veya auto")
    r.add_argument("--no-subtitles", dest="subtitles", action="store_false", help="Altyazı üretme")
    r.add_argument("--single-language", action="store_true",
                   help="Altyazıyı tamamen yayın dilinde üret (varsayılan: dil segment başına tespit edilir; "
                        "yabancı videolar orijinal dilinde yazılır)")
    r.add_argument("--refine-timing", action="store_true",
                   help="Kayan altyazı satırlarını tekrar dinleyerek düzelt (daha doğru senkron, ~4-5 kat yavaş altyazı)")
    r.add_argument("--no-voice-boost", dest="voice_boost", action="store_false",
                   help="Konuşmayı öne çıkarma (vokal ayrıştırıp arka planı kısma) adımını atla")
    r.add_argument("--whisper-model",
                   help="faster-whisper modeli (tiny, base, small, medium, large-v3, large-v3-turbo). Varsayılan: KAC_WHISPER_MODEL veya large-v3")
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
