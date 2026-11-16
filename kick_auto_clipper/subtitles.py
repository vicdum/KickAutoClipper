"""Konuşmayı yazıya döker ve kısa video platformlarına uygun altyazı üretir.

- Konuşma tanıma: faster-whisper (CTranslate2). NVIDIA GPU varsa CUDA, yoksa CPU (int8).
- Kelime düzeyinde zaman damgaları ile kısa (en fazla birkaç kelimelik) altyazı
  parçaları oluşturulur; konuşulan kelime renkle vurgulanır (Shorts/TikTok stili).
- Çıktılar: videoya gömülecek ASS dosyası ve ayrıca yüklenebilecek SRT dosyası.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "large-v3-turbo"

# Whisper'ın sessizlik/müzik üzerinde sık uydurduğu kalıplar (özellikle Türkçe altyazı veri setlerinden).
_HALLUCINATIONS = re.compile(
    r"altyaz[ıi]\s*m\.?\s*k\.?|abone\s+ol|izlediğiniz\s+için\s+teşekkür|"
    r"thanks?\s+for\s+watching|subtitles?\s+by|amara\.org|altyazı\s+çeviri",
    re.IGNORECASE,
)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Cue:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# --- konuşma tanıma ----------------------------------------------------------


def _register_cuda_dlls() -> None:
    """Windows'ta pip ile kurulan nvidia-* paketlerindeki CUDA DLL'lerini yükleme yoluna ekler."""
    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


class Transcriber:
    """Modeli bir kez yükler; GPU'da eşzamanlı çağrılar sıraya alınır."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto"):
        _register_cuda_dlls()
        # Yeni GPU mimarilerinde CUDA çekirdekleri ilk kullanımda derlenir (JIT). Varsayılan
        # 256 MB önbellek Whisper çekirdeklerine yetmeyince her çalıştırmada yeniden derlenir.
        os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 2**30))
        from faster_whisper import WhisperModel

        self.device = device
        if device == "auto":
            try:
                import ctranslate2

                self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                self.device = "cpu"
        compute = "float16" if self.device == "cuda" else "int8"
        try:
            self.model = WhisperModel(model, device=self.device, compute_type=compute)
        except Exception:
            if self.device != "cuda":
                raise
            # CUDA kütüphaneleri eksik/uyumsuzsa CPU'ya düş.
            self.device = "cpu"
            self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.model_name = model
        self._lock = threading.Lock()

    def transcribe(self, media: Path, language: str | None = None, start: float = 0.0, duration: float | None = None) -> list[Word]:
        """`media` dosyasının [start, start+duration] aralığını yazıya döker; zamanlar aralık başına göredir."""
        audio = load_audio(media, start, duration)
        with self._lock:
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,  # tekrar döngülerini ve uydurmaları azaltır
                beam_size=5,
            )
            words: list[Word] = []
            for seg in segments:
                if _HALLUCINATIONS.search(seg.text) or seg.no_speech_prob > 0.8:
                    continue
                for w in seg.words or []:
                    text = w.word.strip()
                    if not text:
                        continue
                    # Boşlukla başlamayan parça önceki kelimenin devamıdır ("Enes" + "'in", "3" + ".000").
                    if words and not w.word.startswith(" "):
                        words[-1].text += text
                        words[-1].end = round(w.end, 3)
                    else:
                        words.append(Word(round(w.start, 3), round(w.end, 3), text))
        return words


def load_audio(media: Path, start: float = 0.0, duration: float | None = None, sr: int = 16000):
    """ffmpeg ile mono 16 kHz float32 ses okur (Whisper'ın beklediği biçim)."""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(media)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


# --- altyazı parçaları -------------------------------------------------------


def build_cues(words: list[Word], max_words: int = 3, max_chars: int = 22, max_gap: float = 0.6) -> list[Cue]:
    """Kelimeleri kısa, okunabilir parçalara böler. Cümle sonu, uzun duraklama,
    kelime veya karakter sınırı yeni parça başlatır."""
    cues: list[Cue] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            text_len = len(" ".join(x.text for x in cur)) + 1 + len(w.text)
            if (
                len(cur) >= max_words
                or text_len > max_chars
                or w.start - cur[-1].end > max_gap
                or cur[-1].text[-1] in ".!?…"
            ):
                cues.append(Cue(cur[0].start, cur[-1].end, cur))
                cur = []
        cur.append(w)
    if cur:
        cues.append(Cue(cur[0].start, cur[-1].end, cur))
    # Parçalar arasında boşluk kısaysa önceki parçayı bir sonrakine kadar ekranda tut (titreşmeyi önler).
    for a, b in zip(cues, cues[1:]):
        if 0 < b.start - a.end < 0.3:
            a.end = b.start
    return cues


def shift(cues: list[Cue], seconds: float) -> list[Cue]:
    """Tüm zamanları `seconds` kadar kaydırılmış yeni parça listesi."""
    if not seconds:
        return cues
    return [
        Cue(c.start + seconds, c.end + seconds, [Word(w.start + seconds, w.end + seconds, w.text) for w in c.words])
        for c in cues
    ]


# --- dosya biçimleri ---------------------------------------------------------


def _ts_srt(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _ts_ass(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def to_srt(cues: list[Cue]) -> str:
    return "".join(f"{i}\n{_ts_srt(c.start)} --> {_ts_srt(c.end)}\n{c.text}\n\n" for i, c in enumerate(cues, 1))


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


@dataclass(frozen=True)
class SubtitleStyle:
    font: str = "Arial Black"
    size_ratio: float = 0.075  # yazı boyutu / video genişliği
    primary: str = "&H00FFFFFF"  # beyaz (ASS: &HAABBGGRR)
    highlight: str = "&H0000E5FF"  # konuşulan kelime: sarı
    outline: str = "&H00000000"
    position: float = 0.72  # yazının dikey konumu (0 üst, 1 alt); blur yerleşiminde alt bulanık alan
    uppercase: bool = True


def upper(text: str, language: str | None = None) -> str:
    """Dile duyarlı büyük harf: Türkçe/Azericede i -> İ, ı -> I."""
    if language in ("tr", "az"):
        text = text.replace("i", "İ")
    return text.upper()


def to_ass(
    cues: list[Cue], width: int, height: int, style: SubtitleStyle = SubtitleStyle(), language: str | None = None
) -> str:
    size = round(width * style.size_ratio)
    margin_v = round(height * (1 - style.position))
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{style.font},{size},{style.primary},{style.primary},{style.outline},&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(2, size // 10)},{max(1, size // 25)},2,{width // 12},{width // 12},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    def tag(color: str) -> str:  # stil rengi &HAABBGGRR -> satır içi etiket \c&HBBGGRR&
        return f"{{\\c&H{color[-6:]}&}}"

    on, off = tag(style.highlight), tag(style.primary)
    lines = []
    for c in cues:
        texts = [_ass_escape(upper(w.text, language) if style.uppercase else w.text) for w in c.words]
        # Parçadaki her kelime konuşulurken ayrı bir olay: o kelime vurgulu, diğerleri beyaz.
        for i, w in enumerate(c.words):
            start = c.start if i == 0 else w.start
            end = c.words[i + 1].start if i + 1 < len(c.words) else c.end
            if end <= start:
                continue
            body = " ".join(f"{on}{t}{off}" if j == i else t for j, t in enumerate(texts))
            lines.append(f"Dialogue: 0,{_ts_ass(start)},{_ts_ass(end)},Caption,,0,0,0,,{body}")
    return header + "\n".join(lines) + "\n"
