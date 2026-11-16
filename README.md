# Kick Auto Clipper

Kick yayınlarındaki viral olabilecek anları sohbet yoğunluğundan tespit edip otomatik klip ve altyazı üreten araç.

## Yol haritası

- [x] **1. İndirme** — zaman damgalı sohbet geçmişi; video yalnızca gereken aralıklar için (varsayılan 1080p)
- [x] **2. Tespit** — sohbetin mesaj/dakika oranının zaman penceresine göre sıçradığı anlar
- [x] **3. Kesme** — sıçramadan 15 sn önce ile oranın tekrar düştüğü andan 15 sn sonrası arası
- [x] **4. Altyazı** — konuşma tanıma ile kelime vurgulu, videoya gömülü altyazı + ayrı SRT

## Kurulum

Gereksinimler: Python 3.10+, [FFmpeg](https://ffmpeg.org/) (PATH'te olmalı; Windows: `winget install Gyan.FFmpeg`).

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[gpu]"      # NVIDIA GPU yoksa: pip install -e .
# Konuşma güçlendirme için PyTorch (NVIDIA GPU, CUDA 12.8):
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

`[gpu]` eki altyazı üretimi için CUDA kütüphanelerini (cuBLAS, cuDNN) kurar. PyTorch kurulu değilse konuşma güçlendirme atlanır; altyazılar etkilenmez. GPU olmayan makinede PyTorch'un CPU sürümü kurulabilir (`pip install torch torchaudio`). İlk çalıştırmada modeller bir kez indirilir: Whisper large-v3 (~3 GB), vokal ayrıştırma (~320 MB).

## Kullanım

```bash
kac run https://kick.com/<kanal>/videos/<uuid>
```

`run` akışı yayının tamamını indirmez:

1. Sohbet geçmişi indirilir (saatlik yayın için ~30 sn).
2. Mesaj/dakika sıçramaları tespit edilir; selamlaşma, vedalaşma ve yayın kesintisi kaynaklı sıçramalar elenir, `clips.json` yazılır.
3. Videodan **yalnızca klip aralıklarına denk gelen HLS segmentleri** indirilir.
4. Aynı segmentlerden konuşma yazıya dökülür ve altyazı, YouTube Shorts / TikTok / Instagram Reels için dikey kodlamayla **tek geçişte** videoya gömülür.

### Kısa video çıktısı

| Özellik | Değer |
| --- | --- |
| En-boy oranı | 9:16 dikey (`--quality 1080` → 1080x1920; düşük kalitede orantılı küçülür, ör. 480 → 480x854) |
| Süre | En fazla 60 sn (`--max-duration`; Shorts/Reels 180, TikTok 600 sn'ye kadar kabul eder) |
| Kodlama | H.264 High, 30 fps, AAC 128k 48 kHz, MP4 `faststart` |
| Boyut | Bit hızı 1080x1920'de 8 Mbps ile sınırlı; 60 sn ≈ 45 MB (TikTok mobil sınırı 287 MB) |
| Yerleşim | `blur`: yatay görüntü ortada, arka planda bulanık kopyası (içerik kaybı yok, varsayılan). `crop`: ortadan 9:16 kesit. `original`: yatay, kodlamasız kopya |
| Kodlayıcı | NVIDIA GPU varsa `h264_nvenc`, yoksa `libx264` (`--encoder`) |

### Altyazı

- **Konuşma tanıma:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper), varsayılan model `large-v3`, dil VOD bilgisinden (ör. `tr`). NVIDIA GPU varsa CUDA (float16), yoksa CPU (int8) kullanılır. Türkçede `large-v3`, `large-v3-turbo`'dan belirgin daha doğru; turbo'da tekrar döngüleri ve uydurma cümleler görüldü.
- **Dil:** Dil her segment için ayrı tespit edilir (`multilingual`). Yayıncılar yabancı videolara tepki verirken dil sabitse Whisper İngilizce konuşmayı bozuk Türkçeye çeviriyor ("Melting pot, bam" → "Yüzüm memnun, bim") ya da tekrar döngüsüne giriyordu ("Evet." ×20); artık orijinal dilinde yazılır. Kısa gürültülü Türkçe bölümler zaman zaman Rusça sanılıp uydurma metin ("Редактор субтитров…") ürettiği için yayın dili ve Latin alfabesi dışında çıkan segmentler yayın dilinde yeniden yazıya dökülür. Türkçe büyük harf kuralı (i → İ) yalnızca Türkçe segmentlere uygulanır. Tamamen yayın dilinde altyazı için `--single-language`.
- **Bağlam:** Klipten önceki 15 sn de modele verilir (sonuçtan çıkarılır). Klip çoğu zaman bir cümlenin ortasında başladığı için bu, karşılaştırmada en büyük doğruluk artışını sağladı (medyan kelime güveni 0,79 → 0,91).
- **Zamanlama:** Kelime zamanları Whisper'ın kendisinden gelir. Senkron, bağımsız bir "tekrar dinleme" testiyle ölçüldü: her altyazı satırının aralığı başka bir modele (large-v3-turbo) tek başına dinletilir ve duyulan metin satırla karşılaştırılır (1 sn yapay kaydırma uyumu 0,61'den 0,22'ye düşürüyor, yani test kaymayı yakalıyor).

  | Yöntem | Uyum | Zayıf satır |
  | --- | --- | --- |
  | **Whisper zamanları (kullanılan)** | **0,61** | **%27** |
  | Zorlamalı hizalama (MMS), tüm klip | 0,56 | %35 |
  | Zorlamalı hizalama (MMS), segment pencereli | 0,41 | %58 |

  Zorlamalı hizalama Türkçe gürültülü yayın sesinde zamanları iyileştirmek yerine bozduğu için kullanılmıyor. VAD sessizlik eşiği (500 ms / 2000 ms / VAD yok) da sonucu değiştirmiyor.
- **Uzun kelimeler:** Whisper kelimelerin ~%5'ine 1-5 sn süre veriyor (kelime yanındaki duraklamayı yutuyor). Bağımsız dinlemeyle ölçüldü: cümle içi kelimelerde konuşma aralığın sonunda, cümlenin ilk kelimesinde başında. Bu kelimeler doğal uzunluğa (~0,35 sn + harf başına 0,09 sn) bu yöne göre kırpılır; böylece satır duraklamada bölünür ve sonraki kelimeler söylenmeden ekrana gelmez. (Kelimelerin yalnızca ~%5'ine dokunduğu için tüm satırlar üzerinden yapılan ölçümde belirgin fark görülmedi.)
- **İsteğe bağlı düzeltme (`--refine-timing`):** Her satır tek başına yeniden dinlenir, metinle uyuşmuyorsa ±1,5 sn içinde en iyi uyduğu konuma kaydırılır. Bağımsız ölçümde kayan satır oranı %8 → %6, ancak altyazı süresi ~4-5 kat uzuyor; bu yüzden varsayılan olarak kapalı.
- **Görünüm:** Kısa video stili; en fazla 3 kelime / 22 karakterlik parçalar, büyük kalın beyaz yazı, siyah kenar çizgisi, konuşulan kelime sarı. Yazı, `blur` yerleşiminde görüntünün hemen altındaki bulanık alanda durur. Türkçe büyük harf kuralı (i → İ) uygulanır.
- **Doğruluk önlemleri:** Sessizlik yalnızca ses etkinliği algılama (VAD) ile belirlenir; Whisper'ın kendi "konuşma yok" kararı kapalıdır, çünkü gürültülü ortamda konuşma olan 30 sn'lik pencerenin tamamını atabiliyor. VAD arka planda şarkı/müzik varken konuşmayı da sessizlik sayabiliyor (bir klipte 59 sn'nin tamamını attı, oysa yayıncı konuşuyordu; vokal ayrıştırılmış seste de aynı). Normal kliplerde VAD sesin %60-90'ını konuşma sayar; %30'undan azını sayarsa klip VAD'sız yeniden yazıya dökülür. Önceki metne koşullama kapalıdır (tekrar döngülerini önler). Ses etiketleri ("Müzik", "[Alkış]", "Gülüşmeler") ve Whisper'ın sessizlikte uydurduğu bilinen kalıplar ("Altyazı M.K.", "abone olmayı unutmayın"…) elenir, parçalı gelen kelimeler birleştirilir ("Enes" + "'in"), noktalama olmasa da cümle sınırında yeni satıra geçilir. Rastgele tohum sabittir.
- **Denenip vazgeçilenler:** Sohbet/başlık kelimeleriyle ipucu (doğruluğu düşürdü, ipucu kelimeleri metne sızdı), ses filtresi (bağlamla birlikte fark yaratmadı), vokal ayrıştırılmış seste tanıma (güven düştü, uydurma arttı), toplu (batched) çıkarım (2 kat hızlı ama kelimelerin ~%30'unu kaçırdı).
- **Çıktı:** Altyazı videoya gömülür (`clip_NN.mp4`); ayrıca platformlara ayrı yüklemek için `clip_NN.srt` yazılır. `--layout original` seçilirse yalnızca SRT üretilir.
- GPU olmayan makinede yerel testleri hızlandırmak için `KAC_WHISPER_MODEL=small` kullanılabilir (doğruluk düşer). Altyazı istemiyorsanız `--no-subtitles`.

### Ses

- **Konuşma güçlendirme:** Klibin sesi vokal ve arka plan olarak ayrıştırılır (HDemucs, torchaudio); vokal olduğu gibi kalır, müzik/masa/kalabalık sesi ~8 dB kısılır. Arka plan tamamen silinmediği için ayrıştırma artefaktları duyulmaz. `--no-voice-boost` ile kapatılabilir.
- **Ses yüksekliği:** 70 Hz altı kesilir, hafif sıkıştırma uygulanır, platformların hedeflediği -14 LUFS'a normalize edilir (ölçülen: -13,3 ile -14,7 LUFS arası).

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
| `--whisper-model` | `large-v3` | Altyazı modeli (yalnızca `run`) |
| `--no-subtitles` | | Altyazı üretme (yalnızca `run`) |
| `--no-voice-boost` | | Konuşma güçlendirmeyi atla (yalnızca `run`) |
| `--refine-timing` | | Kayan altyazı satırlarını tekrar dinleyerek düzelt (yavaş; yalnızca `run`) |
| `--single-language` | | Altyazıyı tamamen yayın dilinde üret (yalnızca `run`) |
| `--refresh-chat` | | Önbellekteki sohbeti yeniden indir |

`kac detect` elenen sıçramaları gerekçesiyle listeler:

```
Elenen: 3 sıçrama
   - 00:00:30-00:01:01  x3.8  açılış
   - 02:58:41-02:58:49  x3.4  az kullanıcı (2)
   - 02:59:42-03:00:33  x6.1  çekiliş anahtar kelimesi (annenpala)
```

Çıktılar `workspace/<kanal>/<vod-uuid>/` altına yazılır:

- `chat.jsonl` — her satır bir mesaj; `t` alanı videonun ilk karesinden itibaren saniyedir
- `clips.json` — tespit edilen anlar (başlangıç/bitiş, zirve anı ve oranı, taban, skor, gürültü oranları); `file_start`, klip dosyasının ilk karesinin video zamanıdır (`original` yerleşiminde anahtar kareye hizalandığı için en fazla ~2 sn önce olabilir)
- `clips/clip_NN_HHMMSS.mp4` — altyazısı gömülü dikey klipler
- `clips/clip_NN_HHMMSS.srt` — klibin altyazısı (zamanlar klip başına göre)
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
| `KAC_WHISPER_MODEL` | `large-v3` | Altyazı modeli (`--whisper-model` her zaman önceliklidir) |
| `KAC_CLIP_WORKERS` | `3` (CPU kodlamada `2`) | Aynı anda hazırlanan klip sayısı |

## Nasıl çalışır

- **VOD bilgisi:** `web.kick.com/api/v1/channels/{kanal_id}/videos/{uuid}` uç noktasından kayıt adresi (HLS `master.m3u8`), başlangıç/bitiş zamanı ve süre alınır.
- **Sohbet:** `kick.com/api/v2/channels/{kanal_id}/messages?cursor=<µs>` verilen andan geriye doğru boşluksuz 25'er mesaj döner. Yayın 60 sn'lik parçalara bölünür, her parça paralel olarak kendi sonundan başına doğru sayfalanır, çakışmalar mesaj kimliğiyle ayıklanır.
- **Zaman hizalama:** Videonun sıfır noktası, oynatma listesindeki ilk segmentin `#EXT-X-PROGRAM-DATE-TIME` değeridir (API'nin `start_time` değerinden birkaç saniye farklı olabilir). Sohbetteki `t` buna göre hesaplanır.
- **Gürültü mesajları:** Her mesaj içeriğine göre sınıflandırılır (`noise.py`): selamlaşma (sa, selam, hg…), vedalaşma (bb, görüşürüz, iyi geceler, kapatma, bye emote'ları…), kesinti tepkisi (drop, f5, yayın gitti, geri geldi, ses yok…) ve bot komutları (`!komut`, bot hesapları). Yalnızca kısa mesajlar (≤6 kelime) gürültü sayılır; Türkçe karakter dönüşümünde karışan kelimeler (dondu/döndü) yalnızca tek başına yazıldığında sayılır.
- **Tespit:** Gürültü mesajları hariç, her kullanıcı 5 sn'de en fazla bir kez sayılarak (spam koruması) 30 sn'lik ortalanmış pencerede mesaj/dk oranı hesaplanır.
  - **Taban çizgisi iki ölçeklidir.** Yerel taban (10 dk kayan medyan) içeriğe göre değişen izleyici sayısını izler; genel taban (tüm yayının medyanı) yerel tabanın aşırı kaymasını sınırlar. Etkin taban ikisinin geometrik ortalamasıdır: yerel taban genelin 4 katına çıkarsa etkin taban 2 katında kalır. Böylece kalabalık bir bölümde gerçek sıçramalar kaybolmaz, sakin bir bölümde küçük dalgalanmalar sıçrama sayılmaz.
  - Oranın `max(2 × etkin taban, 10, anlamlılık eşiği)` değerini aştığı bölgeler aday sıçramadır. 10 sn'den yakın adaylar birleştirilir, 5 sn'den kısalar atlanır.
  - **İstatistiksel anlamlılık:** Sessiz sohbetlerde (dakikada birkaç mesaj) 30 sn'lik pencerede 2-3 mesaj beklenir ve 6 mesaj tesadüfen sıkça görülür. Anlamlılık eşiği, tabanın Poisson dağılımında tesadüfen üretme olasılığı 1/10.000'in altında kalan en düşük orandır: dakikada 5 mesajlık tabanda 4,4 kat, 50 mesajda 1,9 kat, 200 mesajda 1,4 kat. Kalabalık sohbette `2 × taban` kuralı zaten daha katı olduğu için yalnızca sessiz sohbetleri etkiler.
  - **Eleme:** açılış ve kapanıştaki ilk/son 2 dk, oynatma listesinde görülen bir yayın kesintisinden sonraki 90 sn, mesajlarının %30'undan fazlası gürültü ya da %10'undan fazlası kesinti tepkisi olan sıçramalar ve zirvesini 5'ten az kişinin oluşturduğu sıçramalar.
  - **Çekilişler:** Çekilişte bir anahtar kelime duyurulur, izleyiciler katılmak için onu yazar; ardından sonuçlara tepki verilir ("şike", "tüh çıkmadı"). Bunlar viral an değildir. Üç kural:
    1. *Katılım akını:* mesajların %40'ından fazlası en az 8 kişinin tek başına yazdığı aynı kelimeyse (ör. "palavip") ve son 20 dk'da çekilişten bahsedildiyse. Doğal tepki kelimeleri ("gg", "şike", "faul"…) sayılmaz; çekiliş anılmamışsa toplu tek kelime gerçek tepki kabul edilir.
    2. *Çekiliş oturumu:* katılım akınlarının 15 dk öncesinden 5 dk sonrasına kadar olan bölüm (30 dk'dan yakın akınlar aynı oturum).
    3. *Çekiliş anı:* sıçramadan 2 dk önce ile sıçrama sonu arasında en az 3 farklı kişi (bot duyuruları dahil, komut listeleri hariç) çekilişten bahsediyorsa.

  Üç farklı yayında doğrulandı:

  | Yayın | Sohbet | Bulunan | Elenen | Not |
  | --- | --- | --- | --- | --- |
  | muratabigf, 58 dk, bilardo | ~68 msj/dk | 10 | 1 (kapanış) | Kurallar eklenmeden önceki 10 anla aynı |
  | palababayigit, 7,8 sa, oyun + çekiliş | ~13 msj/dk | 0 | 23 | Önce 64 an çıkıyordu; ekran görüntüleriyle hepsinin çekiliş ekranı, geri sayım ya da sohbet dalgalanması olduğu doğrulandı |
  | jahrein, 4,2 sa, sohbet | ~230 msj/dk | 26 | 0 | Gülme (KEKW), korku tepkileri, gündem; yeni kurallar hiçbir anı elemedi |
- **Yayın kesintisi:** Oynatma listesinde `#EXT-X-PROGRAM-DATE-TIME` duvar saati video zamanından 2 sn'den fazla ileri sıçrarsa ya da `#EXT-X-DISCONTINUITY` varsa kesinti sayılır. Kick bağlantı kopunca çoğunlukla yeni bir VOD başlattığı için, bu tür kesintiler genelde VOD başındaki tepki olarak görülür ve açılış kuralı ile gürültü filtresi tarafından yakalanır.
- **Klip indirme:** Kick VOD'ları ~10 sn'lik MPEG-TS segmentlerinden oluşur. Her klip için yalnızca ilgili segmentler paralel indirilir, birleştirilir ve FFmpeg ile kare hassasiyetinde kesilip dikey formata kodlanır.
- Kick Cloudflare arkasında olduğu için istekler `curl_cffi` ile tarayıcı parmak izi taklit edilerek gönderilir.

Örnek ölçüm (58 dk VOD, 10 klip, toplam 9 dk, RTX GPU ile NVENC; sohbet önbellekte):

| Kalite | Klip üretimi | Toplam boyut |
| --- | --- | --- |
| 480p (480x854) | ~13 sn | 100 MB |
| 1080p (1080x1920) | ~77 sn | 383 MB |

Sohbetin ilk indirilişi ~30 sn sürer. CPU kodlamada (libx264) 480p klip üretimi ~48 sn sürdü.

Altyazı ve konuşma güçlendirme ile (480p, `large-v3`, CUDA, RTX 5070 Ti dizüstü): 10 klip 80–300 sn. Klip başına indirme ~1 sn, ses işleme ~2–3 sn, kodlama ~2–3 sn; geri kalanı konuşma tanıma. Konuşma tanıma süresi aynı iş için 4–17 sn arasında değişiyor: dizüstü GPU güç yönetimi, kısa aralıklarla gelen yükte bellek saatini düşürüyor (ölçümde 9001 → 405 MHz). NVIDIA Denetim Masası'nda `python.exe` için "Güç yönetimi modu: Maksimum performansı tercih et" seçmek ve Windows güç planını "Yüksek performans" yapmak bu dalgalanmayı azaltır.

## Testler

```bash
python -m unittest discover -s tests
```

## Lisans

MIT
