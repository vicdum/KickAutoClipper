# Kick Auto Clipper

Kick yayınlarındaki viral olabilecek anları sohbet yoğunluğundan tespit edip otomatik klip ve altyazı üreten araç.

## Yol haritası

- [x] **1. İndirme** — VOD videosu (varsayılan 1080p) ve zaman damgalı sohbet geçmişi
- [ ] **2. Tespit** — sohbetin mesaj/dakika oranının zaman penceresine göre sıçradığı anlar
- [ ] **3. Kesme** — sıçramadan 15 sn önce ile oranın tekrar düştüğü andan 15 sn sonrası arası
- [ ] **4. Altyazı** — kesilen klipler için otomatik altyazı

## Kurulum

Gereksinimler: Python 3.10+, [FFmpeg](https://ffmpeg.org/) (PATH'te olmalı; Windows: `winget install Gyan.FFmpeg`).

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e .
```

## Kullanım

```bash
kac download https://kick.com/<kanal>/videos/<uuid>
```

| Seçenek | Açıklama |
| --- | --- |
| `-q, --quality` | Maksimum yükseklik (1080, 720, 480, 360, 160). Varsayılan 1080 |
| `-o, --workdir` | Çıktı klasörü. Varsayılan `workspace/` |
| `--chat-only` / `--video-only` | Yalnızca sohbeti ya da yalnızca videoyu indir |

Çıktılar `workspace/<kanal>/<vod-uuid>/` altına yazılır:

- `video.mp4` — HLS akışından yeniden kodlamadan alınan video
- `chat.jsonl` — her satır bir mesaj; `t` alanı VOD başından itibaren saniyedir
- `meta.json` — VOD bilgileri (başlangıç/bitiş zamanı, süre, kanal, kalite)

```json
{"t": 1.0, "created_at": "2026-11-06T19:30:13Z", "id": "…", "type": "message", "user_id": 127462302, "user": "zSkyrex31", "content": "!izleme top"}
```

### Yerel ayarlar

`.env.example` dosyasını `.env` olarak kopyalayın. Yerel testlerde işlemleri hızlandırmak için `KAC_QUALITY=480` kullanın. `.env` ve `workspace/` repoya yüklenmez.

| Değişken | Varsayılan | Açıklama |
| --- | --- | --- |
| `KAC_QUALITY` | `1080` | Video çözünürlüğü (`--quality` her zaman önceliklidir) |
| `KAC_WORKDIR` | `workspace` | Çıktı klasörü |
| `KAC_CHAT_WORKERS` | `16` | Sohbet indirmede paralel istek sayısı |

## Nasıl çalışır

- **VOD bilgisi:** `web.kick.com/api/v1/channels/{kanal_id}/videos/{uuid}` uç noktasından kayıt adresi (HLS `master.m3u8`), başlangıç/bitiş zamanı ve süre alınır.
- **Video:** `yt-dlp`, istenen yüksekliği aşmayan en iyi varyantı 16 paralel parça ile indirip MP4'e paketler.
- **Sohbet:** `kick.com/api/v2/channels/{kanal_id}/messages?cursor=<µs>` verilen andan geriye doğru boşluksuz 25'er mesaj döner. Yayın 60 sn'lik parçalara bölünür, her parça paralel olarak kendi sonundan başına doğru sayfalanır, çakışmalar mesaj kimliğiyle ayıklanır.
- Video ve sohbet indirmeleri aynı anda çalışır. Kick Cloudflare arkasında olduğu için istekler `curl_cffi` ile tarayıcı parmak izi taklit edilerek gönderilir.

Örnek ölçüm (58 dk VOD, 480p): video ~40 sn (589 MB), sohbet ~20–30 sn (3968 mesaj).

## Lisans

MIT
