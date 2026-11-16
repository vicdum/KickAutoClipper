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
2. Mesaj/dakika sıçramaları tespit edilir; selamlaşma, vedalaşma ve yayın kesintisi kaynaklı sıçramalar elenir, `clips.json` yazılır.
3. Videodan **yalnızca klip aralıklarına denk gelen HLS segmentleri** indirilir ve YouTube Shorts / TikTok / Instagram Reels için dikey klip olarak kodlanır.

### Kısa video çıktısı

| Özellik | Değer |
| --- | --- |
| En-boy oranı | 9:16 dikey (`--quality 1080` → 1080x1920; düşük kalitede orantılı küçülür, ör. 480 → 480x854) |
| Süre | En fazla 60 sn (`--max-duration`; Shorts/Reels 180, TikTok 600 sn'ye kadar kabul eder) |
| Kodlama | H.264 High, 30 fps, AAC 128k 48 kHz, MP4 `faststart` |
| Boyut | Bit hızı 1080x1920'de 8 Mbps ile sınırlı; 60 sn ≈ 45 MB (TikTok mobil sınırı 287 MB) |
| Yerleşim | `blur`: yatay görüntü ortada, arka planda bulanık kopyası (içerik kaybı yok, varsayılan). `crop`: ortadan 9:16 kesit. `original`: yatay, kodlamasız kopya |
| Kodlayıcı | NVIDIA GPU varsa `h264_nvenc`, yoksa `libx264` (`--encoder`) |

Uzun bir sıçrama süre sınırını aşarsa klip, öncesinde 15 sn bağlam bırakılarak tepkilerin en yoğun olduğu pencereye kırpılır. Dolgu nedeniyle çakışan klipler sınır aşılmıyorsa birleştirilir, aşılıyorsa düşük skorlu olan çakışmayacak şekilde kısaltılır.

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
| `--baseline` | `600` | Yerel taban çizgisi (kayan medyan) penceresi (sn) |
| `--ratio` | `2.0` | Oran, etkin tabanın kaç katını aşınca sıçrama sayılır |
| `--min-rate` | `10` | Sıçrama için minimum mesaj/dk |
| `--pad` | `15` | Sıçramadan önce / düşüşten sonra eklenecek süre (sn) |
| `--intro` / `--outro` | `120` | Yayının ilk / son N saniyesindeki sıçramaları yok say |
| `--max-duration` | `60` | Klip üst sınırı (sn) |
| `--max-clips` | `0` | En yüksek skorlu N klip (0 = hepsi) |
| `--layout` | `blur` | `blur`, `crop` veya `original` (yalnızca `run`) |
| `--encoder` | `auto` | `auto`, `nvenc` veya `cpu` (yalnızca `run`) |
| `--refresh-chat` | | Önbellekteki sohbeti yeniden indir |

`kac detect` elenen sıçramaları gerekçesiyle listeler:

```
Elenen: 1 sıçrama
   - 00:57:07-00:58:28  x4.8  kapanış  (farewell %8)
```

Çıktılar `workspace/<kanal>/<vod-uuid>/` altına yazılır:

- `chat.jsonl` — her satır bir mesaj; `t` alanı videonun ilk karesinden itibaren saniyedir
- `clips.json` — tespit edilen anlar (başlangıç/bitiş, zirve anı ve oranı, taban, skor, gürültü oranları); `file_start`, klip dosyasının ilk karesinin video zamanıdır (`original` yerleşiminde anahtar kareye hizalandığı için en fazla ~2 sn önce olabilir)
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
| `KAC_ENCODER` | `auto` | Video kodlayıcı (`--encoder` her zaman önceliklidir) |

## Nasıl çalışır

- **VOD bilgisi:** `web.kick.com/api/v1/channels/{kanal_id}/videos/{uuid}` uç noktasından kayıt adresi (HLS `master.m3u8`), başlangıç/bitiş zamanı ve süre alınır.
- **Sohbet:** `kick.com/api/v2/channels/{kanal_id}/messages?cursor=<µs>` verilen andan geriye doğru boşluksuz 25'er mesaj döner. Yayın 60 sn'lik parçalara bölünür, her parça paralel olarak kendi sonundan başına doğru sayfalanır, çakışmalar mesaj kimliğiyle ayıklanır.
- **Zaman hizalama:** Videonun sıfır noktası, oynatma listesindeki ilk segmentin `#EXT-X-PROGRAM-DATE-TIME` değeridir (API'nin `start_time` değerinden birkaç saniye farklı olabilir). Sohbetteki `t` buna göre hesaplanır.
- **Gürültü mesajları:** Her mesaj içeriğine göre sınıflandırılır (`noise.py`): selamlaşma (sa, selam, hg…), vedalaşma (bb, görüşürüz, iyi geceler, kapatma, bye emote'ları…), kesinti tepkisi (drop, f5, yayın gitti, geri geldi, ses yok…) ve bot komutları (`!komut`, bot hesapları). Yalnızca kısa mesajlar (≤6 kelime) gürültü sayılır; Türkçe karakter dönüşümünde karışan kelimeler (dondu/döndü) yalnızca tek başına yazıldığında sayılır.
- **Tespit:** Gürültü mesajları hariç, her kullanıcı 5 sn'de en fazla bir kez sayılarak (spam koruması) 30 sn'lik ortalanmış pencerede mesaj/dk oranı hesaplanır.
  - **Taban çizgisi iki ölçeklidir.** Yerel taban (10 dk kayan medyan) içeriğe göre değişen izleyici sayısını izler; genel taban (tüm yayının medyanı) yerel tabanın aşırı kaymasını sınırlar. Etkin taban ikisinin geometrik ortalamasıdır: yerel taban genelin 4 katına çıkarsa etkin taban 2 katında kalır. Böylece kalabalık bir bölümde gerçek sıçramalar kaybolmaz, sakin bir bölümde küçük dalgalanmalar sıçrama sayılmaz.
  - Oranın `max(2 × etkin taban, 10)` eşiğini aştığı bölgeler aday sıçramadır. 10 sn'den yakın adaylar birleştirilir, 5 sn'den kısalar atlanır.
  - **Eleme:** açılış ve kapanıştaki ilk/son 2 dk, oynatma listesinde görülen bir yayın kesintisinden sonraki 90 sn, mesajlarının %30'undan fazlası gürültü ya da %10'undan fazlası kesinti tepkisi olan sıçramalar.
- **Yayın kesintisi:** Oynatma listesinde `#EXT-X-PROGRAM-DATE-TIME` duvar saati video zamanından 2 sn'den fazla ileri sıçrarsa ya da `#EXT-X-DISCONTINUITY` varsa kesinti sayılır. Kick bağlantı kopunca çoğunlukla yeni bir VOD başlattığı için, bu tür kesintiler genelde VOD başındaki tepki olarak görülür ve açılış kuralı ile gürültü filtresi tarafından yakalanır.
- **Klip indirme:** Kick VOD'ları ~10 sn'lik MPEG-TS segmentlerinden oluşur. Her klip için yalnızca ilgili segmentler paralel indirilir, birleştirilir ve FFmpeg ile kare hassasiyetinde kesilip dikey formata kodlanır.
- Kick Cloudflare arkasında olduğu için istekler `curl_cffi` ile tarayıcı parmak izi taklit edilerek gönderilir.

Örnek ölçüm (58 dk VOD, 10 klip, toplam 9 dk, RTX GPU ile NVENC; sohbet önbellekte):

| Kalite | Klip üretimi | Toplam boyut |
| --- | --- | --- |
| 480p (480x854) | ~13 sn | 100 MB |
| 1080p (1080x1920) | ~77 sn | 383 MB |

Sohbetin ilk indirilişi ~30 sn sürer. CPU kodlamada (libx264) 480p klip üretimi ~48 sn sürdü.

## Testler

```bash
python -m unittest discover -s tests
```

## Lisans

MIT
