"""Konuşmayı öne çıkaran ses işleme (klibin izleyiciye giden sesi için).

Vokal ayrıştırma (HDemucs, torchaudio) ile yayın sesindeki konuşma; müzik, oyun sesi
ve ortam gürültüsünden ayrılır, arka plan kısılarak yeniden karıştırılır.

Not: Kelime zamanları için zorlamalı hizalama (MMS_FA) da denendi. Bağımsız bir
"tekrar dinleme" testinde (her altyazı satırının aralığı Whisper'a tek başına
dinletilip metin karşılaştırıldı) Whisper'ın kendi zamanlarından kötü çıktı
(uyum 0,61 -> 0,41; tüm klip boyunca hizalama 0,56), bu yüzden kullanılmıyor.

PyTorch gerektirir; yoksa bu adım atlanır.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

_lock = threading.Lock()

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
