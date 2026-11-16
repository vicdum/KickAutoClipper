"""Sohbet yoğunluğundan viral olabilecek anları tespit eder.

Yöntem:
1. Gürültü mesajları (selamlaşma, vedalaşma, kesinti tepkileri, bot komutları)
   sayımdan çıkarılır; aynı kullanıcı `user_cooldown` saniyede en fazla bir kez
   sayılır (spam tek kişinin sıçrama üretmesini engeller).
2. Anlık oran: `window` saniyelik ortalanmış pencerede mesaj/dakika.
3. Taban çizgisi iki ölçeklidir: yerel taban (`baseline_window` saniyelik kayan
   medyan) izleyici sayısındaki değişimi izler; genel taban (tüm yayının medyanı)
   yerel tabanın içerikle birlikte aşırı kaymasını sınırlar. Etkin taban ikisinin
   geometrik ortalamasıdır.
4. Oranın `max(ratio * taban, min_rate)` eşiğini aştığı bölgeler aday sıçramadır.
5. Adaylar reddedilir: açılış/kapanış bölgesindeyse, bir yayın kesintisinin hemen
   ardındaysa ya da mesajlarının büyük kısmı gürültüyse.
6. Klip = [yükselme - pad_before, düşüş + pad_after]; `max_duration` saniyeyi
   aşan klipler tepkinin en yoğun olduğu pencereye kırpılır (kısa video platformları).
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, insort
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from statistics import median

from .noise import NOISE_KINDS, TECHNICAL, classify


@dataclass
class DetectParams:
    window: int = 30  # anlık oran penceresi (sn)
    baseline_window: int = 600  # yerel taban penceresi (sn)
    ratio: float = 2.0  # eşik = ratio * etkin taban
    min_rate: float = 10.0  # mesaj/dk; sessiz yayınlarda gürültüyü eler
    significance: float = 1e-4  # sıçramanın tesadüf olma olasılığı üst sınırı (Poisson)
    min_users: int = 5  # sıçramaya katılan en az farklı kullanıcı
    keyword_share: float = 0.4  # aynı tek kelimenin oranı bunu aşarsa çekiliş katılımı olabilir
    keyword_users: int = 8  # ...ve en az bu kadar farklı kişi yazdıysa
    giveaway_lookback: float = 1200.0  # çekiliş anılmasının aranacağı geçmiş süre (sn)
    giveaway_before: float = 900.0  # çekiliş oturumu: ilk katılım akınından bu kadar önce başlar
    giveaway_after: float = 300.0  # ... son akından bu kadar sonra biter
    giveaway_users: int = 3  # sıçrama çevresinde çekilişten bu kadar kişi bahsederse çekiliş anı sayılır
    merge_gap: int = 10  # bu kadar yakın sıçramalar tek an sayılır (sn)
    min_spike: int = 5  # daha kısa sıçramalar yok sayılır (sn)
    user_cooldown: float = 5.0  # bir kullanıcının tekrar sayılması için gereken süre (sn)
    intro: float = 120.0  # yayının ilk N saniyesindeki sıçramalar yok sayılır
    outro: float = 120.0  # yayının son N saniyesindeki sıçramalar yok sayılır
    interruption_guard: float = 90.0  # kesintiden sonraki N saniye yok sayılır
    max_noise: float = 0.3  # sıçramadaki gürültü mesajı oranı bunu aşarsa reddedilir
    max_technical: float = 0.1  # kesinti tepkisi oranı bunu aşarsa reddedilir
    pad_before: float = 15.0
    pad_after: float = 15.0
    max_duration: float = 60.0  # Shorts/Reels/TikTok için klip üst sınırı (sn)
    min_duration: float = 15.0  # kırpma/çakışma sonrası bundan kısa klipler atılır
    max_clips: int = 0  # 0 = sınırsız; aksi halde en yüksek skorlu N klip


@dataclass
class Clip:
    start: float  # VOD başından saniye
    end: float
    spike_start: int  # oranın eşiği aştığı an
    spike_end: int  # oranın eşiğin altına indiği an
    peak: int  # oranın zirve yaptığı an
    peak_rate: float  # mesaj/dk (gürültü hariç)
    baseline: float  # zirve anındaki etkin taban (mesaj/dk)
    score: float  # peak_rate / taban
    noise: dict[str, float] = field(default_factory=dict)  # sıçramadaki gürültü oranları

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {**asdict(self), "duration": round(self.duration, 3)}


@dataclass
class Rejected:
    clip: Clip
    reason: str


@dataclass
class DetectResult:
    clips: list[Clip]
    rejected: list[Rejected]


# --- sinyal hazırlama -------------------------------------------------------


def signal_counts(messages: list[dict], duration: float, cooldown: float) -> tuple[list[int], list[str | None]]:
    """Saniyelik sayımlar (gürültü hariç, kullanıcı başına bekleme süreli) ve mesaj sınıfları."""
    n = int(duration) + 1
    counts = [0] * n
    kinds: list[str | None] = []
    last_seen: dict = {}
    for m in messages:
        kind = classify(m.get("content", ""), m.get("user"))
        kinds.append(kind)
        i = int(m["t"])
        if kind is not None or not 0 <= i < n:
            continue
        uid = m.get("user_id") or m.get("user")
        if uid is not None and m["t"] - last_seen.get(uid, -math.inf) < cooldown:
            continue
        last_seen[uid] = m["t"]
        counts[i] += 1
    return counts, kinds


def prefix_sums(values: list[int]) -> list[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + v)
    return out


def rolling_rate(counts: list[int], window: int) -> list[float]:
    """Ortalanmış pencerede mesaj/dakika."""
    n, prefix, half = len(counts), prefix_sums(counts), window // 2
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


def effective_baseline(local: list[float], global_median: float) -> list[float]:
    """Yerel ve genel tabanın geometrik ortalaması; biri sıfırsa diğeri."""
    out = []
    for l in local:
        out.append(math.sqrt(l * global_median) if l > 0 and global_median > 0 else max(l, global_median))
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


# --- kırpma ve birleştirme ---------------------------------------------------


def _best_window(c: Clip, length: float, prefix: list[int], pad_before: float) -> tuple[float, float]:
    """Klip içinde `length` uzunluğunda, tepkinin (öncesinde `pad_before` sn bağlam
    bırakarak) en yoğun olduğu pencereyi seçer."""
    lo, hi = int(c.start), int(c.end - length)
    n = len(prefix) - 1

    def reactions(s: int) -> int:
        a, b = min(n, s + int(pad_before)), min(n, s + int(length))
        return prefix[b] - prefix[a]

    best = max(range(lo, max(lo, hi) + 1), key=lambda s: (reactions(s), -abs(s - lo)))
    return float(best), float(best) + length


def _fit(clips: list[Clip], p: DetectParams, prefix: list[int]) -> list[Clip]:
    for c in clips:
        if c.duration > p.max_duration:
            c.start, c.end = _best_window(c, p.max_duration, prefix, p.pad_before)

    out: list[Clip] = []
    for c in sorted(clips, key=lambda c: c.start):
        if out and c.start < out[-1].end:
            prev = out[-1]
            if max(prev.end, c.end) - prev.start <= p.max_duration:
                best = prev if prev.score >= c.score else c
                out[-1] = Clip(
                    start=prev.start, end=max(prev.end, c.end),
                    spike_start=prev.spike_start, spike_end=max(prev.spike_end, c.spike_end),
                    peak=best.peak, peak_rate=best.peak_rate, baseline=best.baseline,
                    score=best.score, noise=best.noise,
                )
                continue
            # Birleşince sınır aşılıyor: düşük skorlu klibi çakışmayacak şekilde kısalt.
            if prev.score >= c.score:
                c.start = prev.end
            else:
                prev.end = c.start
        out.append(c)
    return [c for c in out if c.duration >= p.min_duration]


# --- çekiliş anahtar kelimesi ------------------------------------------------

# Çekilişlerde bir anahtar kelime duyurulur, izleyiciler katılmak için onu sohbete yazar.
# Bu toplu yazım sohbet sıçraması üretir ama viral an değildir.
_GIVEAWAY = re.compile(r"[çc]ekili[şs]|anahtar\s*kel[iİı]me|giveaway|raffle|keyword|[öo]d[üu]l|kazanan", re.IGNORECASE)
_EMOTE_ONLY = re.compile(r"^(\[emote:\d+:[^\]]+\]\s*)+$")
# Doğal tepki kelimeleri anahtar kelime sayılmaz (toplu "gg" gerçek bir tepkidir).
_REACTIONS = {"gg", "ez", "w", "l", "xd", "lol", "lmao", "kekw", "omg", "wow", "sike", "faul", "gol", "ole",
              "aaa", "hahaha", "ahaha", "yuh", "oha", "vay", "evet", "hayir", "hadi", "abi", "hoca"}


def _is_command_text(content: str) -> bool:
    """Bot komutu ya da komut listesi: '!' ile başlar veya en az 3 '!komut' içerir."""
    content = content.strip()
    return content.startswith("!") or len(re.findall(r"(?:^|\s)![a-zA-Z0-9]", content)) >= 3


def giveaway_keyword(messages: list[dict], p: DetectParams) -> str | None:
    """Mesajların büyük kısmı, çok sayıda kişinin tek başına yazdığı aynı kelimeyse o kelimeyi döner."""
    from .noise import normalize

    texts: dict[str, set] = {}
    total = 0
    for m in messages:
        content = m.get("content", "").strip()
        if not content or _EMOTE_ONLY.match(content):
            continue
        total += 1
        tokens = normalize(content).split()
        if len(tokens) == 1:
            texts.setdefault(tokens[0], set()).add(m.get("user_id") or m.get("user"))
    if not texts or not total:
        return None
    word, users = max(texts.items(), key=lambda kv: len(kv[1]))
    count = sum(1 for m in messages if normalize(m.get("content", "")).split() == [word])
    if word in _REACTIONS or len(users) < p.keyword_users or count / max(len(messages), 1) < p.keyword_share:
        return None
    return word


def giveaway_sessions(flood_times: list[int], p: DetectParams, join_gap: float = 1800.0) -> list[tuple[float, float]]:
    """Katılım akınlarını (30 dk'dan yakın olanlar aynı oturum) oturum aralıklarına çevirir."""
    sessions: list[list[float]] = []
    for t in sorted(flood_times):
        if sessions and t - sessions[-1][1] <= join_gap:
            sessions[-1][1] = t
        else:
            sessions.append([t, t])
    return [(a - p.giveaway_before, b + p.giveaway_after) for a, b in sessions]


# --- istatistiksel anlamlılık -------------------------------------------------


@lru_cache(maxsize=4096)
def _poisson_quantile(mu_x10: int, p: float) -> int:
    """Ortalaması mu olan Poisson dağılımında P(X >= k) <= p olan en küçük k."""
    mu = mu_x10 / 10
    if mu <= 0:
        return 1
    pmf = math.exp(-mu)
    cdf = pmf
    k = 0
    while 1 - cdf > p:
        k += 1
        pmf *= mu / k
        cdf += pmf
    return k + 1


def significant_rate(base_rate: float, window: int, p: float) -> float:
    """Tabanın tesadüfen üretemeyeceği en düşük oran (mesaj/dk).

    Sessiz sohbetlerde (dakikada birkaç mesaj) 30 sn'lik pencerede 2-3 mesaj beklenir ve
    6 mesaj tesadüfen sıkça görülür; oran eşiği (taban x 2) bu dalgalanmayı sıçrama sayar.
    Poisson eşiği az mesajda çok daha büyük bir sıçrama ister, kalabalık sohbette ise
    taban x 2 kuralı zaten daha katı olduğu için etkisizdir.
    """
    mu = base_rate * window / 60
    return _poisson_quantile(round(mu * 10), p) * 60 / window


# --- ana fonksiyon -----------------------------------------------------------


def detect(
    messages: list[dict],
    duration: float,
    p: DetectParams | None = None,
    interruptions: list[tuple[float, float]] | None = None,
) -> DetectResult:
    """`messages`: `t` (sn), `content`, `user`, `user_id` alanlı sohbet mesajları."""
    p = p or DetectParams()
    counts, kinds = signal_counts(messages, duration, p.user_cooldown)
    rate = rolling_rate(counts, p.window)
    local = rolling_median(rate, p.baseline_window)
    active = [r for r in rate if r > 0]
    base = effective_baseline(local, median(active) if active else 0.0)
    flags = [r >= max(p.ratio * b, p.min_rate, significant_rate(b, p.window, p.significance))
             for r, b in zip(rate, base)]

    merged: list[list[int]] = []
    for s, e in _regions(flags):
        if merged and s - merged[-1][1] < p.merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    times = [m["t"] for m in messages]
    # Çekilişten bahseden mesajlar: izleyiciler ve bot duyuruları ("BÜYÜK ÇEKİLİŞ BAŞLIYOR").
    # Komut listeleri ("!8ball !ai !cekilis ...") ve tek komutlar ("!cekilis") sayılmaz.
    mentions = [(m["t"], m.get("user_id") or m.get("user")) for m in messages
                if _GIVEAWAY.search(m.get("content", "")) and not _is_command_text(m.get("content", ""))]
    giveaway_times = [t for t, _ in mentions]
    clips: list[Clip] = []
    rejected: list[Rejected] = []
    for s, e in merged:
        if e - s < p.min_spike:
            continue
        peak = max(range(s, e), key=rate.__getitem__)
        b = max(base[peak], 1e-9)
        lo, hi = bisect_left(times, s), bisect_left(times, e)
        tally = Counter(kinds[lo:hi])
        total = max(hi - lo, 1)
        noise = {k: round(tally[k] / total, 3) for k in NOISE_KINDS if tally[k]}
        # Zirveyi oluşturan kişi sayısı: zirve penceresi ile sıçrama bölgesinin kesişimindeki farklı
        # kullanıcılar (bölge, ortalanmış pencere yüzünden asıl patlamadan geniş; kısa sıçramalarda
        # ise pencere bölgeden geniş olduğundan ikisinin kesişimi alınır).
        plo = bisect_left(times, max(s, peak - p.window / 2))
        phi = bisect_left(times, min(e, peak + p.window / 2))
        users = len({messages[j].get("user_id") or messages[j].get("user") for j in range(plo, phi) if kinds[j] is None})
        clip = Clip(
            start=max(0.0, s - p.pad_before),
            end=min(float(duration), e + p.pad_after),
            spike_start=s, spike_end=e, peak=peak,
            peak_rate=round(rate[peak], 2), baseline=round(base[peak], 2),
            score=round(rate[peak] / b, 2), noise=noise,
        )
        reason = _reject_reason(clip, duration, p, interruptions or [])
        if not reason and users < p.min_users:
            reason = f"az kullanıcı ({users})"
        if not reason:
            kw = giveaway_keyword(messages[lo:hi], p)
            g = bisect_left(giveaway_times, s - p.giveaway_lookback)
            if kw and g < len(giveaway_times) and giveaway_times[g] <= e:
                reason = f"çekiliş anahtar kelimesi ({kw})"
        if not reason:
            g0, g1 = bisect_left(giveaway_times, s - 120), bisect_left(giveaway_times, e)
            if len({u for _, u in mentions[g0:g1]}) >= p.giveaway_users:
                reason = "çekiliş"
        if reason:
            rejected.append(Rejected(clip, reason))
        else:
            clips.append(clip)

    # Çekiliş oturumu: onaylanmış katılım akınlarının çevresi. Bu sürede ekranda çekiliş aracı
    # vardır ve sohbet çekiliş sonuçlarına tepki verir ("şike", "tüh çıkmadı"); viral an değildir.
    sessions = giveaway_sessions([r.clip.spike_start for r in rejected if r.reason.startswith("çekiliş anahtar")], p)
    kept = []
    for c in clips:
        if any(a <= c.spike_start <= b for a, b in sessions):
            rejected.append(Rejected(c, "çekiliş oturumu"))
        else:
            kept.append(c)
    clips = kept
    rejected.sort(key=lambda r: r.clip.spike_start)

    if p.max_clips > 0:
        clips = sorted(clips, key=lambda c: c.score, reverse=True)[: p.max_clips]
    return DetectResult(_fit(clips, p, prefix_sums(counts)), rejected)


def _reject_reason(c: Clip, duration: float, p: DetectParams, interruptions: list[tuple[float, float]]) -> str | None:
    if c.spike_start < p.intro:
        return "açılış"
    if c.spike_end > duration - p.outro:
        return "kapanış"
    for t, _ in interruptions:
        if t <= c.spike_start < t + p.interruption_guard:
            return "yayın kesintisi"
    if c.noise.get(TECHNICAL, 0) > p.max_technical:
        return "kesinti tepkisi"
    if sum(c.noise.values()) > p.max_noise:
        top = max(c.noise, key=c.noise.get)
        return {"greeting": "selamlaşma", "farewell": "vedalaşma", "command": "bot komutları"}.get(top, top)
    return None
