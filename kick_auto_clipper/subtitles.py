"""Konuşmayı yazıya döker ve kısa video platformlarına uygun altyazı üretir.

- Konuşma tanıma: faster-whisper (CTranslate2). NVIDIA GPU varsa CUDA, yoksa CPU (int8).
  Klipten önceki 15 sn bağlam olarak verilir; kelime zamanları zorlamalı hizalamayla
  düzeltilir (bkz. `speech.py`).
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

from . import speech as _speech

# Türkçe dahil İngilizce dışı dillerde large-v3, turbo sürümünden belirgin daha doğru
# (turbo'da kod çözücü katmanları 32'den 4'e indirilmiştir).
DEFAULT_MODEL = "large-v3"

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
    sentence_end: bool = False  # Whisper segmentinin son kelimesi (noktalama olmasa da cümle sınırı)


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

    def transcribe(
        self,
        media: Path,
        language: str | None = None,
        start: float = 0.0,
        duration: float | None = None,
        context: float = 15.0,
        align: bool = True,
    ) -> list[Word]:
        """`media` dosyasının [start, start+duration] aralığını yazıya döker; zamanlar `start`'a göredir.

        - `context`: aralıktan önceki bu kadar saniye de modele verilir. Klip bir cümlenin
          ortasında başlayabildiği için bağlam tanıma doğruluğunu belirgin artırır; bu
          bölümdeki kelimeler sonuçtan çıkarılır.
        - `align`: PyTorch kuruluysa kelime zamanları zorlamalı hizalamayla düzeltilir
          (Whisper'ın 1-2 sn'lik kaymalarını giderir).
        """
        ctx_start = max(0.0, start - context)
        pre = start - ctx_start
        audio = load_audio(media, ctx_start, None if duration is None else duration + pre)
        with self._lock:
            # Whisper zor bölümlerde rastgele örneklemeye (sıcaklık > 0) geçer; tohum sabitlenmezse
            # aynı ses her çalıştırmada farklı metin üretebilir.
            import ctranslate2

            ctranslate2.set_random_seed(0)
            result, _ = self.model.transcribe(
                audio,
                language=language,
                # Kelime zamanları hizalamada düzeltilse de kapatılmamalı: Whisper bir sonraki
                # 30 sn'lik pencereye nereden devam edeceğini bunlarla belirler; kapalıyken
                # pencereler atlanıp metnin bir kısmı kaybolabiliyor.
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,  # tekrar döngülerini ve uydurmaları azaltır
                # Sessizlik tespiti yalnızca VAD ile yapılır. Whisper'ın kendi "konuşma yok" kararı
                # (no_speech_prob > 0.6 ve düşük güven) gürültülü ortamda konuşma olan 30 sn'lik
                # pencerenin tamamını atabiliyor; bir klipte 35 sn'lik konuşma bu yüzden kayboldu.
                # Aynı nedenle hallucination_silence_threshold da kullanılmıyor. Uydurmalara karşı
                # VAD ve bilinen kalıp filtresi yeterli.
                no_speech_threshold=None,
                beam_size=5,
            )
            segments = [_merge_pieces(s) for s in result if not _HALLUCINATIONS.search(s.text)]
        segments = [s for s in segments if s[2]]

        if align and _speech.available():
            timed = _speech.align_segments(audio, segments)
            segments = [(s0, s1, [(a, b, w[2]) for (a, b), w in zip(t, ws)]) for (s0, s1, ws), t in zip(segments, timed)]

        words = []
        for _, _, ws in segments:
            for k, (a, b, text) in enumerate(ws):
                if b - pre > 0.05:  # bağlam bölümündeki kelimeleri at
                    words.append(Word(round(max(a - pre, 0.0), 3), round(b - pre, 3), text, k == len(ws) - 1))
        return words


def _merge_pieces(seg) -> tuple[float, float, list[tuple[float, float, str]]]:
    """Segmentin kelimelerini (başlangıç, bitiş, metin) olarak döner. Boşlukla başlamayan
    parça önceki kelimenin devamıdır ("Enes" + "'in", "3" + ".000")."""
    words: list[list] = []
    for w in seg.words or []:
        text = w.word.strip()
        if not text:
            continue
        if words and not w.word.startswith(" "):
            words[-1][1] = w.end
            words[-1][2] += text
        else:
            words.append([w.start, w.end, text])
    return seg.start, seg.end, [tuple(w) for w in words]


def load_audio(media: Path, start: float = 0.0, duration: float | None = None, sr: int = 16000):
    """ffmpeg ile mono 16 kHz float32 ses okur (Whisper'ın beklediği biçim)."""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(media)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def load_stereo(media: Path, start: float, duration: float, sr: int = 44100):
    """ffmpeg ile (2, N) float32 stereo ses okur."""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(media), "-t", f"{duration:.3f}",
           "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def write_wav(path: Path, stereo, sr: int) -> None:
    """(2, N) float32 sesi 16-bit PCM WAV olarak yazar."""
    import wave

    import numpy as np

    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)


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
                or cur[-1].sentence_end
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
        Cue(c.start + seconds, c.end + seconds,
            [Word(w.start + seconds, w.end + seconds, w.text, w.sentence_end) for w in c.words])
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
