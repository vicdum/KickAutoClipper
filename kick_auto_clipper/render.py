"""Kısa video platformları (YouTube Shorts, TikTok, Instagram Reels) için çıktı ayarları.

Ortak gereksinimler: 9:16 dikey, H.264 + AAC, MP4 (faststart), 30 fps.
- 1080x1920 tüm platformların önerdiği çözünürlüktür; daha düşük kalite seçilirse
  (ör. yerel testlerde 480) en-boy oranı korunarak küçültülür.
- Bit hızı üst sınırı, 60 sn'lik bir klibi ~50 MB civarında tutar (TikTok'un
  mobil yükleme sınırı 287 MB, diğerlerinin sınırları çok daha yüksek).
- Süre sınırı tespit aşamasında uygulanır (`DetectParams.max_duration`).

Yerleşimler:
- blur: yatay görüntü tam genişlikte ortada, arka planda aynı görüntünün
  bulanıklaştırılmış hali. İçerik kaybı olmaz; yayın klipleri için güvenli varsayılan.
- crop: görüntünün ortasından 9:16 kesit. Ekranı doldurur, kenarlar kaybolur.
- original: yatay, yeniden kodlamasız kopya (anahtar kareye hizalı kesim).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache

LAYOUTS = ("blur", "crop", "original")

# Konuşma odaklı ses: 70 Hz altındaki uğultuyu kes, dinamikleri hafifçe sıkıştır (fısıltı ile
# bağırma arası farkı azaltır), platformların hedeflediği -14 LUFS ses yüksekliğine normalize et.
AUDIO_FILTER = (
    "highpass=f=70,"
    "acompressor=threshold=-21dB:ratio=3:attack=5:release=120:makeup=2,"
    "loudnorm=I=-14:TP=-1.5:LRA=11"
)
ENCODERS = ("auto", "nvenc", "cpu")


@lru_cache(maxsize=None)
def nvenc_available(ffmpeg: str) -> bool:
    """NVIDIA donanım kodlayıcısı bu makinede gerçekten çalışıyor mu (kısa deneme kodlaması)."""
    try:
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_encoder(choice: str, ffmpeg: str) -> str:
    if choice == "auto":
        return "nvenc" if nvenc_available(ffmpeg) else "cpu"
    return choice


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


@dataclass(frozen=True)
class OutputFormat:
    layout: str = "blur"
    width: int = 1080  # dikey çıktının genişliği (kısa kenar)
    encoder: str = "cpu"  # cpu (libx264) | nvenc (NVIDIA GPU)
    fps: int = 30
    crf: int = 21
    preset: str = "veryfast"
    audio_bitrate: str = "128k"

    @property
    def height(self) -> int:
        return _even(self.width * 16 / 9)

    @property
    def maxrate_kbps(self) -> int:
        # 1080x1920 için 8 Mbps; piksel sayısıyla orantılı küçülür.
        return max(1000, int(8000 * (self.width * self.height) / (1080 * 1920)))

    @classmethod
    def for_quality(cls, quality: int, layout: str = "blur", encoder: str = "cpu") -> "OutputFormat":
        if layout not in LAYOUTS:
            raise ValueError(f"Bilinmeyen yerleşim: {layout} (seçenekler: {', '.join(LAYOUTS)})")
        return cls(layout=layout, width=_even(min(quality, 1080)), encoder=encoder)

    def video_codec_args(self) -> list[str]:
        rate = ["-maxrate", f"{self.maxrate_kbps}k", "-bufsize", f"{2 * self.maxrate_kbps}k"]
        if self.encoder == "nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                    "-cq", str(self.crf + 2), "-b:v", "0", *rate, "-profile:v", "high"]
        return ["-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
                *rate, "-profile:v", "high", "-level", "4.1"]

    def filter_complex(self, subtitles: str | None = None) -> str:
        """`subtitles`: çalışma klasöründeki ASS dosyasının adı (videoya gömülür)."""
        w, h = self.width, self.height
        subs = f",ass={subtitles}" if subtitles else ""
        if self.layout == "crop":
            return f"[0:v]scale=-2:{h},crop={w}:{h},setsar=1,fps={self.fps}{subs},format=yuv420p[v]"
        # blur: arka plan küçük çözünürlükte bulanıklaştırılıp büyütülür (hızlı).
        bw, bh = _even(w / 6), _even(h / 6)
        return (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"boxblur=8:2,scale={w}:{h},setsar=1[bgb];"
            f"[fg]scale={w}:-2,setsar=1[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={self.fps}{subs},format=yuv420p[v]"
        )

    def ffmpeg_args(self, subtitles: str | None = None, external_audio: bool = False) -> list[str]:
        """Girdi sonrası ffmpeg argümanları (çıktı yolu hariç). `external_audio`: ses ikinci
        girdiden (ör. konuşması güçlendirilmiş WAV) alınır."""
        if self.layout == "original":
            return ["-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart"]
        return [
            "-filter_complex", self.filter_complex(subtitles),
            "-map", "[v]", "-map", "1:a:0" if external_audio else "0:a:0?",
            *self.video_codec_args(),
            "-g", str(self.fps * 2),
            "-af", AUDIO_FILTER,
            "-c:a", "aac", "-b:a", self.audio_bitrate, "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
        ]
