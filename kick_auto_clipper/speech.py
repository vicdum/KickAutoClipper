"""Konuşma odaklı ses işleme: vokal ayrıştırma ve kelime zamanlarının hizalanması.

- Vokal ayrıştırma (HDemucs, torchaudio): yayın sesindeki müzik, oyun sesi ve ortam
  gürültüsünü konuşmadan ayırır. Konuşma tanıma bu temiz ses üzerinde çalışır.
- Zorlamalı hizalama (MMS_FA, torchaudio): Whisper'ın kelime zaman damgaları
  dikkat ağırlıklarından tahmin edildiği için gürültülü seste 1-2 sn kayabilir.
  Tanınan metin, ses üzerinde karakter düzeyinde CTC ile yeniden hizalanır; kelime
  başlangıç/bitişleri ~20 ms çözünürlükle belirlenir.

Her iki model de PyTorch gerektirir (`pip install -e ".[gpu]"`); yoksa bu adımlar atlanır.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

_lock = threading.Lock()

# MMS_FA sözlüğü yalnızca Latin a-z ve kesme işareti içerir; Türkçe karakterler eşlenir.
_ROMAN = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i", "û": "u",
                        "ä": "a", "é": "e", "è": "e", "à": "a", "ñ": "n", "ß": "s", "’": "'"})


def available() -> bool:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=None)
def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --- vokal ayrıştırma ---------------------------------------------------------


@lru_cache(maxsize=None)
def _demucs():
    import torchaudio

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(_device()).eval()
    return model, bundle.sample_rate


def isolate_vocals(stereo: np.ndarray, sr: int, segment: float = 10.0, overlap: float = 1.0) -> np.ndarray:
    """(2, N) float32 stereo sesten vokal kanalını ayırır; aynı biçimde döner.

    Model uzun sesi tek seferde işleyemeyeceği için `segment` saniyelik, `overlap`
    saniye örtüşen parçalarla çalışılır; örtüşmeler doğrusal geçişle birleştirilir.
    """
    import torch
    import torchaudio.functional as AF

    model, model_sr = _demucs()
    vocals_idx = model.sources.index("vocals")
    x = torch.from_numpy(stereo).float()
    if sr != model_sr:
        x = AF.resample(x, sr, model_sr)
    n = x.shape[1]
    seg, hop = int(segment * model_sr), int((segment - overlap) * model_sr)
    fade = int(overlap * model_sr)
    out = torch.zeros_like(x)
    weight = torch.zeros(n)
    ramp = torch.linspace(0, 1, fade)
    with _lock, torch.inference_mode():
        for start in range(0, max(n - fade, 1), hop):
            chunk = x[:, start : start + seg]
            # Model girişini normalize et (torchaudio örneğiyle aynı).
            mean, std = chunk.mean(), chunk.std() + 1e-8
            y = model(((chunk - mean) / std).unsqueeze(0).to(_device()))[0, vocals_idx].cpu() * std + mean
            w = torch.ones(y.shape[1])
            if start > 0:
                w[:fade] = ramp[: min(fade, len(w))]
            if start + seg < n:
                w[-fade:] = torch.minimum(w[-fade:], ramp.flip(0)[: len(w[-fade:])])
            out[:, start : start + y.shape[1]] += y * w
            weight[start : start + y.shape[1]] += w
    out /= weight.clamp(min=1e-6)
    if sr != model_sr:
        out = AF.resample(out, model_sr, sr)
    return out.numpy()


def voice_boost(stereo: np.ndarray, sr: int, background_gain: float = 0.4) -> np.ndarray:
    """Konuşmayı öne çıkarır: vokal olduğu gibi kalır, geri kalan (müzik, oyun, ortam)
    `background_gain` ile kısılır (0.4 ≈ -8 dB). Arka plan tamamen silinmediği için
    ayrıştırma artefaktları duyulmaz ve klibin atmosferi korunur."""
    vocals = isolate_vocals(stereo, sr)
    n = min(vocals.shape[1], stereo.shape[1])
    mix = vocals[:, :n] + background_gain * (stereo[:, :n] - vocals[:, :n])
    peak = float(np.abs(mix).max()) or 1.0
    return (mix / peak * 0.98 if peak > 0.98 else mix).astype(np.float32)


# --- zorlamalı hizalama -------------------------------------------------------


@lru_cache(maxsize=None)
def _aligner():
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(_device()).eval()
    return model, bundle.get_tokenizer(), bundle.get_aligner(), bundle.sample_rate, set(bundle.get_dict())


def romanize(word: str, vocab: set[str]) -> str:
    w = word.replace("İ", "i").replace("I", "ı").lower().translate(_ROMAN)
    return "".join(ch for ch in w if ch in vocab and ch != "-")


def align(audio16k: np.ndarray, words: list[str]) -> list[tuple[float, float, float] | None]:
    """Her kelime için (başlangıç, bitiş, skor) döner; hizalanamayan (ör. yalnızca rakam) kelimeler için None."""
    import torch

    model, tokenizer, aligner, sr, vocab = _aligner()
    assert sr == 16000
    keys = [romanize(w, vocab) for w in words]
    idx = [i for i, k in enumerate(keys) if k]
    result: list[tuple[float, float, float] | None] = [None] * len(words)
    if not idx:
        return result
    wav = torch.from_numpy(audio16k).float().unsqueeze(0)
    with _lock, torch.inference_mode():
        emission, _ = model(wav.to(_device()))
        spans = aligner(emission[0], tokenizer([keys[i] for i in idx]))
    sec_per_frame = wav.shape[1] / emission.shape[1] / sr
    for i, sp in zip(idx, spans):
        score = float(sum(s.score * (s.end - s.start) for s in sp) / max(sum(s.end - s.start for s in sp), 1))
        result[i] = (sp[0].start * sec_per_frame, sp[-1].end * sec_per_frame, score)
    return result


def align_segments(
    audio16k: np.ndarray,
    segments: list[tuple[float, float, list[tuple[float, float, str]]]],
    margin: float = 2.0,
) -> list[list[tuple[float, float]]]:
    """Her Whisper segmentini kendi zaman penceresinde ([başlangıç - margin, bitiş + margin]) hizalar.

    Tüm sesi tek seferde hizalamak, yanlış tanınmış bir kelimenin komşularını saniyelerce
    uzağa çekmesine yol açabilir; segment düzeyindeki zamanlar kelime düzeyindekilerden
    çok daha güvenilir olduğu için pencere hataları sınırlar (WhisperX yaklaşımı).
    `segments`: (başlangıç, bitiş, [(kelime_başlangıç, kelime_bitiş, metin), ...]).
    Dönen liste her segment için kelime (başlangıç, bitiş) zamanlarıdır.
    """
    out = []
    n = len(audio16k)
    prev_end = 0.0
    for s_start, s_end, words in segments:
        fallback = [(w[0], w[1]) for w in words]
        # Whisper segmentleri örtüşebilir: pencere önceki segmentin son kelimesinden önce başlamaz.
        lo = max(0, int(max(s_start - margin, prev_end) * 16000))
        hi = min(n, int((s_end + margin) * 16000))
        try:
            if hi - lo < 1600:  # 0.1 sn'den kısa
                raise ValueError("pencere çok kısa")
            offset = lo / 16000
            times = [(t[0] + offset, t[1] + offset, t[2]) if t else None
                     for t in align(audio16k[lo:hi], [w[2] for w in words])]
            seg_times = fill_gaps(times, fallback)
        except Exception:
            seg_times = fallback  # hizalanamazsa Whisper zamanları
        seg_times = monotonic(seg_times, prev_end)
        out.append(seg_times)
        if seg_times:
            prev_end = seg_times[-1][1]
    return out


def monotonic(times: list[tuple[float, float]], floor: float = 0.0) -> list[tuple[float, float]]:
    """Kelime zamanlarını sıralı hale getirir: hiçbir kelime öncekinden önce başlamaz."""
    out = []
    for a, b in times:
        a = max(a, floor)
        b = max(b, a)
        out.append((a, b))
        floor = b
    return out


def fill_gaps(times: list[tuple[float, float, float] | None], fallback: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Hizalanamayan kelimelere komşu kelimeler arasında, Whisper süresine orantılı zaman verir."""
    out: list[tuple[float, float] | None] = [(t[0], t[1]) if t else None for t in times]
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        lo = out[i - 1][1] if i > 0 else fallback[i][0]
        hi = out[j][0] if j < len(out) else fallback[j - 1][1]
        if hi < lo:
            hi = lo
        total = sum(max(fallback[k][1] - fallback[k][0], 0.05) for k in range(i, j))
        t = lo
        for k in range(i, j):
            d = (hi - lo) * max(fallback[k][1] - fallback[k][0], 0.05) / total
            out[k] = (t, t + d)
            t += d
        i = j
    return [o for o in out if o is not None]
