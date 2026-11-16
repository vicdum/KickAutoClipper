"""Kick web API istemcisi.

Kick, Cloudflare arkasında olduğu için istekler curl_cffi ile tarayıcı
TLS parmak izi taklit edilerek gönderilir.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from curl_cffi import requests

WEB_API = "https://web.kick.com/api/v1"
LEGACY_API = "https://kick.com/api"
IMPERSONATE = "chrome"

_VOD_URL = re.compile(
    r"kick\.com/(?P<slug>[\w-]+)/videos/(?P<uuid>[0-9a-f-]{36})", re.IGNORECASE
)

_local = threading.local()


class KickError(RuntimeError):
    pass


def session() -> requests.Session:
    """İş parçacığı başına bir oturum (curl_cffi oturumları thread-safe değildir)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session(impersonate=IMPERSONATE, timeout=30)
        _local.session = s
    return s


def get_json(url: str, params: dict | None = None, retries: int = 5) -> dict:
    """GET + JSON; 429/5xx ve ağ hatalarında üstel geri çekilme ile yeniden dener."""
    delay = 1.0
    for attempt in range(retries):
        try:
            r = session().get(url, params=params)
        except requests.RequestsError as e:
            if attempt == retries - 1:
                raise KickError(f"{url}: {e}") from e
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                raise KickError(f"Bulunamadı: {url}")
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise KickError(f"{url}: HTTP {r.status_code}")
        time.sleep(delay)
        delay *= 2
    raise KickError(f"{url}: yeniden deneme sınırı aşıldı")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class Vod:
    uuid: str
    title: str
    channel_id: int
    channel_slug: str
    start_time: str  # ISO 8601, UTC
    end_time: str
    duration: float  # saniye
    playlist_url: str  # HLS master.m3u8
    language: str | None = None  # yayın dili (ör. "tr")

    @property
    def start(self) -> datetime:
        return parse_iso(self.start_time)

    @property
    def end(self) -> datetime:
        return parse_iso(self.end_time)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_vod_url(url: str) -> tuple[str, str]:
    m = _VOD_URL.search(url)
    if not m:
        raise KickError(f"Desteklenmeyen VOD adresi: {url} (beklenen: kick.com/<kanal>/videos/<uuid>)")
    return m["slug"].lower(), m["uuid"].lower()


def get_channel(slug: str) -> dict:
    return get_json(f"{LEGACY_API}/v2/channels/{slug}")


def get_vod(url: str) -> Vod:
    slug, uuid = parse_vod_url(url)
    channel = get_channel(slug)
    data = get_json(f"{WEB_API}/channels/{channel['id']}/videos/{uuid}")["data"]
    if not data.get("recording_url"):
        raise KickError("VOD için kayıt adresi (recording_url) bulunamadı; video hâlâ işleniyor olabilir.")
    return Vod(
        uuid=data["id"],
        title=data.get("title", ""),
        channel_id=data["channel"]["id"],
        channel_slug=data["channel"]["slug"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        duration=float(data["duration"]),
        playlist_url=data["recording_url"],
        language=data.get("language"),
    )
