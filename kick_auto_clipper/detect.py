"""Sohbet yoğunluğundan viral olabilecek anları tespit eder.

Yöntem:
1. Mesajlar saniyelik kutulara sayılır.
2. Anlık oran: `window` saniyelik ortalanmış pencerede mesaj/dakika.
3. Taban çizgisi: anlık oranın `baseline_window` saniyelik kayan medyanı
   (medyan, sıçramalardan etkilenmediği için yayının doğal akışını izler).
4. Oranın `max(ratio * taban, min_rate)` eşiğini aştığı kesintisiz bölgeler
   sıçrama sayılır; aralarında `merge_gap` saniyeden az boşluk olanlar birleşir.
5. Klip = [yükselme - pad_before, düşüş + pad_after]; çakışan klipler birleşir.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import asdict, dataclass
from statistics import median


@dataclass
class DetectParams:
    window: int = 30  # anlık oran penceresi (sn)
    baseline_window: int = 600  # taban çizgisi penceresi (sn)
    ratio: float = 2.0  # eşik = ratio * taban
    min_rate: float = 10.0  # mesaj/dk; sessiz yayınlarda gürültüyü eler
    merge_gap: int = 10  # bu kadar yakın sıçramalar tek an sayılır (sn)
    min_duration: int = 5  # daha kısa sıçramalar yok sayılır (sn)
    pad_before: float = 15.0
    pad_after: float = 15.0
    max_clips: int = 0  # 0 = sınırsız; aksi halde en yüksek skorlu N klip


@dataclass
class Clip:
    start: float  # VOD başından saniye
    end: float
    spike_start: int  # oranın eşiği aştığı an
    spike_end: int  # oranın eşiğin altına indiği an
    peak_rate: float  # mesaj/dk
    baseline: float  # zirve anındaki taban (mesaj/dk)
    score: float  # peak_rate / taban

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {**asdict(self), "duration": round(self.duration, 3)}


def per_second_counts(times: list[float], duration: float) -> list[int]:
    n = int(duration) + 1
    counts = [0] * n
    for t in times:
        i = int(t)
        if 0 <= i < n:
            counts[i] += 1
    return counts


def rolling_rate(counts: list[int], window: int) -> list[float]:
    """Ortalanmış pencerede mesaj/dakika."""
    n = len(counts)
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append((prefix[hi] - prefix[lo]) * 60.0 / (hi - lo))
    return out


def rolling_median(values: list[float], window: int, step: int = 5) -> list[float]:
    """Ortalanmış kayan medyan; `step` saniyede bir hesaplanıp aradaki değerler doldurulur."""
    n = len(values)
    if n == 0:
        return []
    half = window // 2
    out = [0.0] * n
    sorted_win: list[float] = []
    lo = hi = 0  # sorted_win, values[lo:hi] içerir
    for i in range(0, n, step):
        new_lo, new_hi = max(0, i - half), min(n, i + half + 1)
        while hi < new_hi:
            insort(sorted_win, values[hi])
            hi += 1
        while lo < new_lo:
            sorted_win.pop(bisect_left(sorted_win, values[lo]))
            lo += 1
        m = median(sorted_win)
        for j in range(i, min(i + step, n)):
            out[j] = m
    return out


def _regions(flags: list[bool]) -> list[tuple[int, int]]:
    regions, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(flags)))
    return regions


def detect(times: list[float], duration: float, p: DetectParams | None = None) -> list[Clip]:
    p = p or DetectParams()
    counts = per_second_counts(times, duration)
    rate = rolling_rate(counts, p.window)
    base = rolling_median(rate, p.baseline_window)
    flags = [r >= max(p.ratio * b, p.min_rate) for r, b in zip(rate, base)]

    # Yakın sıçramaları birleştir, kısa olanları ele.
    merged: list[list[int]] = []
    for s, e in _regions(flags):
        if merged and s - merged[-1][1] < p.merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    spikes = [(s, e) for s, e in merged if e - s >= p.min_duration]

    clips: list[Clip] = []
    for s, e in spikes:
        peak_i = max(range(s, e), key=rate.__getitem__)
        b = max(base[peak_i], 1e-9)
        clips.append(
            Clip(
                start=max(0.0, s - p.pad_before),
                end=min(float(duration), e + p.pad_after),
                spike_start=s,
                spike_end=e,
                peak_rate=round(rate[peak_i], 2),
                baseline=round(base[peak_i], 2),
                score=round(rate[peak_i] / b, 2),
            )
        )

    if p.max_clips > 0:
        clips = sorted(clips, key=lambda c: c.score, reverse=True)[: p.max_clips]
        clips.sort(key=lambda c: c.start)

    # Dolgu nedeniyle çakışan klipleri birleştir.
    out: list[Clip] = []
    for c in clips:
        if out and c.start <= out[-1].end:
            prev = out[-1]
            best = prev if prev.score >= c.score else c
            out[-1] = Clip(
                start=prev.start,
                end=max(prev.end, c.end),
                spike_start=prev.spike_start,
                spike_end=max(prev.spike_end, c.spike_end),
                peak_rate=best.peak_rate,
                baseline=best.baseline,
                score=best.score,
            )
        else:
            out.append(c)
    return out
