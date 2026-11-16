"""VOD sohbet geçmişini zaman damgalarıyla indirir.

Kick'in sohbet geçmişi uç noktası `cursor` (mikrosaniye cinsinden Unix zamanı)
ile çağrıldığında o andan *geriye doğru* boşluksuz 25'er mesaj döner ve bir
sonraki (daha eski) imleci verir. Yayın süresini parçalara bölüp her parçayı
kendi sonundan başlangıcına doğru paralel sayfalayarak tüm sohbet hızlıca
toplanır; parça sınırlarındaki çakışmalar mesaj kimliğiyle ayıklanır.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

from .kick_api import LEGACY_API, Vod, get_json, parse_iso

PAGE_URL = LEGACY_API + "/v2/channels/{channel_id}/messages"


def _to_cursor(t: datetime) -> str:
    return str(int(t.timestamp() * 1_000_000))


def _fetch_segment(channel_id: int, seg_start: datetime, seg_end: datetime) -> list[dict]:
    """[seg_start, seg_end) aralığındaki mesajları geriye doğru sayfalayarak toplar."""
    url = PAGE_URL.format(channel_id=channel_id)
    cursor = _to_cursor(seg_end)
    out: list[dict] = []
    while True:
        data = get_json(url, params={"cursor": cursor}).get("data") or {}
        messages = data.get("messages") or []
        out.extend(messages)
        next_cursor = data.get("cursor")
        if not messages or not next_cursor or int(next_cursor) >= int(cursor):
            break
        if min(parse_iso(m["created_at"]) for m in messages) < seg_start:
            break
        cursor = next_cursor
    return out


def _segments(start: datetime, end: datetime, length: timedelta) -> Iterable[tuple[datetime, datetime]]:
    t = start
    while t < end:
        yield t, min(t + length, end)
        t += length


def _normalize(m: dict, vod_start: datetime) -> dict:
    created = parse_iso(m["created_at"])
    sender = m.get("sender") or {}
    return {
        "t": round((created - vod_start).total_seconds(), 3),  # VOD başından itibaren saniye
        "created_at": m["created_at"],
        "id": m["id"],
        "type": m.get("type"),
        "user_id": m.get("user_id"),
        "user": sender.get("username"),
        "content": m.get("content", ""),
    }


def download_chat(
    vod: Vod,
    out_path: Path,
    workers: int = 16,
    segment_seconds: int = 60,
    progress: Callable[[int, int], None] | None = None,
    zero: datetime | None = None,
) -> int:
    """VOD sohbetini `out_path` dosyasına JSONL olarak yazar, mesaj sayısını döner.

    `zero`, videonun ilk karesinin duvar saatidir (HLS PROGRAM-DATE-TIME). Verilirse
    `t` alanı buna göre hesaplanır; yoksa API'nin `start_time` değeri kullanılır.
    """
    zero = zero or vod.start
    start = min(zero, vod.start)
    end = max(vod.end, zero + timedelta(seconds=vod.duration)) + timedelta(seconds=1)
    segments = list(_segments(start, end, timedelta(seconds=segment_seconds)))

    by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_segment, vod.channel_id, s, e) for s, e in segments]
        for done, fut in enumerate(as_completed(futures), 1):
            for m in fut.result():
                by_id[m["id"]] = m
            if progress:
                progress(done, len(futures))

    rows = [
        _normalize(m, zero)
        for m in by_id.values()
        if start <= parse_iso(m["created_at"]) < end
    ]
    rows.sort(key=lambda r: (r["t"], r["id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return len(rows)


def load_chat(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
