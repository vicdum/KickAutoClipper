# Kick Auto Clipper

Kick yayınlarındaki viral olabilecek anları sohbet yoğunluğundan tespit edip otomatik klip ve altyazı üreten araç.

## Yol haritası

- [x] **1. İndirme** — zaman damgalı sohbet geçmişi; video yalnızca gereken aralıklar için (varsayılan 1080p)
- [x] **2. Tespit** — sohbetin mesaj/dakika oranının zaman penceresine göre sıçradığı anlar
- [x] **3. Kesme** — sıçramadan 15 sn önce ile oranın tekrar düştüğü andan 15 sn sonrası arası
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
kac run https://kick.com/<kanal>/videos/<uuid>
```

`run` akışı yayının tamamını indirmez:

1. Sohbet geçmişi indirilir (saatlik yayın için ~30 sn).
2. Mesaj/dakika sıçramaları tespit edilir, `clips.json` yazılır.
3. Videodan **yalnızca klip aralıklarına denk gelen HLS segmentleri** indirilir ve yeniden kodlamadan kesilir.

| Komut | Ne yapar |
| --- | --- |
| `kac run <url>` | Sohbet → tespit → klipler |
| `kac detect <url>` | Yalnızca tespit; video indirmez. Parametre ayarlamak için (sohbet önbellekten okunur) |
| `kac download <url>` | Videonun tamamını ve sohbeti indirir |

| Seçenek | Varsayılan | Açıklama |
| --- | --- | --- |
| `-q, --quality` | `1080` | Maksimum yükseklik (1080, 720, 480, 360, 160) |
| `-o, --workdir` | `workspace` | Çıktı klasörü |
| `--window` | `30` | Anlık mesaj/dk oranının hesaplandığı pencere (sn) |
| `--baseline` | `600` | Taban çizgisi (kayan medyan) penceresi (sn) |
| `--ratio` | `2.0` | Oran, tabanın kaç katını aşınca sıçrama sayılır |
| `--min-rate` | `10` | Sıçrama için minimum mesaj/dk |
| `--pad` | `15` | Sıçramadan önce / düşüşten sonra eklenecek süre (sn) |
| `--max-clips` | `0` | En yüksek skorlu N klip (0 = hepsi) |
| `--refresh-chat` | | Önbellekteki sohbeti yeniden indir |

Çıktılar `workspace/<kanal>/<vod-uuid>/` altına yazılır:

- `chat.jsonl` — her satır bir mesaj; `t` alanı videonun ilk karesinden itibaren saniyedir
- `clips.json` — tespit edilen anlar (başlangıç/bitiş, zirve oranı, taban, skor); `file_start`, klip dosyasının ilk karesinin video zamanıdır (anahtar kareye hizalı, istenen başlangıçtan en fazla ~2 sn önce)
- `clips/clip_NN_HHMMSS.mp4` — kesilen klipler
- `meta.json` — VOD bilgileri
- `video.mp4` — yalnızca `kac download` ile

```json
{"t": 1.457, "created_at": "2026-11-06T19:30:11Z", "id": "…", "type": "message", "user_id": 127462302, "user": "zSkyrex31", "content": "!izleme top"}
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
- **Sohbet:** `kick.com/api/v2/channels/{kanal_id}/messages?cursor=<µs>` verilen andan geriye doğru boşluksuz 25'er mesaj döner. Yayın 60 sn'lik parçalara bölünür, her parça paralel olarak kendi sonundan başına doğru sayfalanır, çakışmalar mesaj kimliğiyle ayıklanır.
- **Zaman hizalama:** Videonun sıfır noktası, oynatma listesindeki ilk segmentin `#EXT-X-PROGRAM-DATE-TIME` değeridir (API'nin `start_time` değerinden birkaç saniye farklı olabilir). Sohbetteki `t` buna göre hesaplanır.
- **Tespit:** Mesajlar saniyelik sayılır; 30 sn'lik ortalanmış pencerede mesaj/dk oranı hesaplanır. Taban çizgisi bu oranın 10 dk'lık kayan medyanıdır, böylece yayının doğal akışı izlenirken sıçramalar tabanı bozmaz. Oranın `max(2 × taban, 10)` eşiğini aştığı bölgeler sıçramadır. 10 sn'den yakın sıçramalar birleştirilir, 5 sn'den kısalar elenir, dolgu sonrası çakışan klipler birleştirilir.
- **Klip indirme:** Kick VOD'ları ~10 sn'lik MPEG-TS segmentlerinden oluşur. Her klip için yalnızca ilgili segmentler paralel indirilir, birleştirilir ve FFmpeg ile yeniden kodlamadan (stream copy) kesilir.
- Kick Cloudflare arkasında olduğu için istekler `curl_cffi` ile tarayıcı parmak izi taklit edilerek gönderilir.

Örnek ölçüm (58 dk VOD, 480p): sohbet ~30 sn (3968 mesaj), tespit <1 sn, 10 klip (13 dk, 123 MB) ~5 sn. Toplam ~35 sn. Karşılaştırma için videonun tamamını indirmek ~40 sn ve 589 MB sürüyor; aradaki fark yayın uzadıkça ve çözünürlük arttıkça büyür.

## Testler

```bash
python -m unittest discover -s tests
```

## Lisans

MIT
