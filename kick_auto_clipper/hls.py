"""Kick VOD HLS akışları: varyant seçimi, segment zaman çizelgesi ve aralık indirme.

Kick VOD'ları ~10 sn'lik MPEG-TS segmentlerinden oluşur. Bir klip için yalnızca
o aralığa denk gelen segmentler paralel indirilir, uç uca eklenir ve ffmpeg ile
yeniden kodlanmadan (stream copy) kesilir. Böylece saatlerce sürebilen yayının
tamamını indirmek gerekmez.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from .kick_api import parse_iso, session


@dataclass
class Variant:
    url: str
    height: int
    bandwidth: int


@dataclass
class Segment:
    url: str
    start: float  # video başından saniye
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class MediaPlaylist:
    url: str
    height: int
    segments: list[Segment] = field(default_factory=list)
    program_date_time: datetime | None = None  # ilk segmentin duvar saati (UTC)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def segments_between(self, start: float, end: float) -> list[Segment]:
        return [s for s in self.segments if s.end > start and s.start < end]


def _get_text(url: str) -> str:
    r = session().get(url)
    r.raise_for_status()
    return r.text


def _attrs(line: str) -> dict[str, str]:
    out, key, buf, quoted = {}, "", "", False
    for ch in line.split(":", 1)[1] + ",":
        if ch == '"':
            quoted = not quoted
        elif ch == "=" and not quoted and not key:
            key, buf = buf, ""
        elif ch == "," and not quoted:
            if key:
                out[key.strip()] = buf
            key, buf = "", ""
        else:
            buf += ch
    return out


def parse_master(text: str, base_url: str) -> list[Variant]:
    variants, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF:"):
            a = _attrs(line)
            height = int(a.get("RESOLUTION", "0x0").split("x")[1])
            uri = next(l for l in lines[i + 1 :] if l and not l.startswith("#"))
            variants.append(Variant(urljoin(base_url, uri), height, int(a.get("BANDWIDTH", 0))))
    return variants


def choose_variant(variants: list[Variant], max_height: int) -> Variant:
    fitting = [v for v in variants if v.height <= max_height]
    pool = fitting or [min(variants, key=lambda v: v.height)]
    return max(pool, key=lambda v: (v.height, v.bandwidth))


def parse_media(text: str, url: str, height: int) -> MediaPlaylist:
    pl = MediaPlaylist(url=url, height=height)
    t, dur = 0.0, None
    for line in text.splitlines():
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:") and pl.program_date_time is None:
            pl.program_date_time = parse_iso(line.split(":", 1)[1].strip())
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and dur is not None:
            pl.segments.append(Segment(urljoin(url, line), t, dur))
            t += dur
            dur = None
    return pl


def load_playlist(master_url: str, max_height: int) -> MediaPlaylist:
    variant = choose_variant(parse_master(_get_text(master_url), master_url), max_height)
    return parse_media(_get_text(variant.url), variant.url, variant.height)


def _fetch(url: str) -> bytes:
    for attempt in range(5):
        try:
            r = session().get(url)
            if r.status_code == 200:
                return r.content
        except Exception:
            if attempt == 4:
                raise
    raise RuntimeError(f"Segment indirilemedi: {url}")


def _keyframe_before(ts_path: Path, offset: float, ffprobe: str) -> float:
    """`offset` (dosya başından sn) anına eşit veya önceki son anahtar karenin konumu."""
    r = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
            "-show_entries", "frame=pts_time:format=start_time", "-of", "json", str(ts_path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(r.stdout)
    t0 = float(info["format"]["start_time"])
    keys = [float(f["pts_time"]) - t0 for f in info.get("frames", []) if "pts_time" in f]
    before = [k for k in keys if k <= offset + 1e-3]
    return max(before) if before else 0.0


def download_range(
    pl: MediaPlaylist,
    start: float,
    end: float,
    out_path: Path,
    ffmpeg: str = "ffmpeg",
    workers: int = 8,
) -> float:
    """[start, end) aralığını (video saniyesi) MP4 olarak kaydeder.

    Yeniden kodlama yapılmadığı için kesim, `start` anından önceki son anahtar
    kareye çekilir (Kick'te ~2 sn arayla). Klibin gerçek başlangıcını döner.
    """
    segs = pl.segments_between(start, end)
    if not segs:
        raise ValueError(f"Aralık videonun dışında: {start}-{end}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(_fetch, [s.url for s in segs]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".ts", dir=out_path.parent, delete=False) as tmp:
        for c in chunks:
            tmp.write(c)
        joined = Path(tmp.name)
    try:
        ffprobe = str(Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe")))
        offset = _keyframe_before(joined, max(0.0, start - segs[0].start), ffprobe)
        actual_start = segs[0].start + offset
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{offset:.3f}", "-i", str(joined),
                "-t", f"{end - actual_start:.3f}",
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(out_path),
            ],
            check=True,
        )
    finally:
        joined.unlink(missing_ok=True)
    return actual_start
