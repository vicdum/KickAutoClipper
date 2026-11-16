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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from .kick_api import KickError, parse_iso, session


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
    # Yayın kesintileri: (video saniyesi, kayıp süre sn). Kesinti, duvar saatinin
    # video zaman çizelgesinden ileri sıçramasından veya DISCONTINUITY etiketinden anlaşılır.
    interruptions: list[tuple[float, float]] = field(default_factory=list)

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


def parse_media(text: str, url: str, height: int, gap_tolerance: float = 2.0) -> MediaPlaylist:
    pl = MediaPlaylist(url=url, height=height)
    t, dur, drift = 0.0, None, 0.0
    discontinuity = False
    for line in text.splitlines():
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pdt = parse_iso(line.split(":", 1)[1].strip())
            if pl.program_date_time is None:
                pl.program_date_time = pdt
            else:
                # Duvar saati ile video zamanı arasındaki fark büyüdüyse yayın kesilmiş demektir.
                new_drift = (pdt - pl.program_date_time).total_seconds() - t
                if new_drift - drift > gap_tolerance:
                    pl.interruptions.append((round(t, 3), round(new_drift - drift, 3)))
                    discontinuity = False
                drift = new_drift
        elif line.startswith("#EXT-X-DISCONTINUITY") and not line.startswith("#EXT-X-DISCONTINUITY-"):
            discontinuity = True
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and dur is not None:
            if discontinuity and pl.segments:
                pl.interruptions.append((round(t, 3), 0.0))
            discontinuity = False
            pl.segments.append(Segment(urljoin(url, line), t, dur))
            t += dur
            dur = None
    return pl


def load_playlist(master_url: str, max_height: int) -> MediaPlaylist:
    variant = choose_variant(parse_master(_get_text(master_url), master_url), max_height)
    return parse_media(_get_text(variant.url), variant.url, variant.height)


def _fetch(url: str, retries: int = 6) -> bytes:
    """Segment indirir; ağ hatası ve 5xx/429'da üstel geri çekilmeyle (~1+2+4+8+16 sn) yeniden dener."""
    delay, last = 1.0, None
    for attempt in range(retries):
        try:
            r = session().get(url)
            if r.status_code == 200:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # curl_cffi ağ hataları
            last = str(e)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    raise KickError(f"Segment indirilemedi ({last}): {url}")


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


@dataclass
class RangeFile:
    """Bir aralığı kapsayan, uç uca eklenmiş segment dosyası."""

    path: Path
    file_start: float  # dosyanın ilk karesinin video zamanı (sn)

    def offset(self, t: float) -> float:
        """Video zamanını dosya içi zamana çevirir."""
        return max(0.0, t - self.file_start)


def fetch_range(pl: MediaPlaylist, start: float, end: float, work_dir: Path, workers: int = 8) -> RangeFile:
    """[start, end) aralığını kapsayan segmentleri paralel indirip tek .ts dosyasında birleştirir.
    Dosyayı silmek çağıranın sorumluluğundadır."""
    segs = pl.segments_between(start, end)
    if not segs:
        raise ValueError(f"Aralık videonun dışında: {start}-{end}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(_fetch, [s.url for s in segs]))
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".ts", dir=work_dir, delete=False) as tmp:
        for c in chunks:
            tmp.write(c)
    return RangeFile(Path(tmp.name), segs[0].start)


def cut(
    src: RangeFile,
    start: float,
    end: float,
    out_path: Path,
    ffmpeg: str = "ffmpeg",
    encode_args: list[str] | None = None,
    extra_inputs: list[Path] | None = None,
) -> float:
    """`src` içinden [start, end) aralığını (video saniyesi) MP4 olarak kaydeder, klibin gerçek başlangıcını döner.

    `encode_args` verilirse yeniden kodlanır ve kesim kare hassasiyetindedir. ffmpeg,
    `out_path` klasöründe çalıştırılır; böylece filtrelerde (ör. `ass=`) dosya adları
    kaçış karakteri gerektirmeden kullanılabilir. Verilmezse yeniden kodlamadan kopyalanır;
    bu durumda kesim `start` anından önceki son anahtar kareye çekilir (Kick'te ~2 sn arayla).
    """
    offset = src.offset(start)
    if encode_args is None:
        ffprobe = str(Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe")))
        offset = _keyframe_before(src.path, offset, ffprobe)
        encode_args = ["-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart"]
    actual_start = src.file_start + offset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{offset:.3f}", "-i", str(src.path.resolve()),
            *[arg for p in extra_inputs or [] for arg in ("-i", str(p.resolve()))],
            "-t", f"{end - actual_start:.3f}",
            *encode_args,
            out_path.name,
        ],
        check=True,
        cwd=out_path.parent,
    )
    return actual_start
