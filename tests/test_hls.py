import unittest

from kick_auto_clipper.hls import choose_variant, parse_master, parse_media

MASTER = """#EXTM3U
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=4364908,CODECS="avc1.4D0028,mp4a.40.2",RESOLUTION=1920x1080,VIDEO="1080p",FRAME-RATE=30.000
1080p/playlist.m3u8
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=2399241,CODECS="avc1.4D401F,mp4a.40.2",RESOLUTION=1280x720,VIDEO="720p30",FRAME-RATE=30.000
720p30/playlist.m3u8
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=1454241,CODECS="avc1.4D401F,mp4a.40.2",RESOLUTION=852x480,VIDEO="480p30",FRAME-RATE=30.000
480p30/playlist.m3u8
"""

MEDIA = """#EXTM3U
#EXT-X-TARGETDURATION:12
#EXT-X-PROGRAM-DATE-TIME:2026-11-06T19:30:09.543Z
#EXTINF:10.000,
0.ts
#EXT-X-PROGRAM-DATE-TIME:2026-11-06T19:30:19.543Z
#EXTINF:10.000,
1.ts
#EXTINF:3.334,
2.ts
#EXT-X-ENDLIST
"""

BASE = "https://stream.example/media/hls/master.m3u8"


class HlsTest(unittest.TestCase):
    def test_master_and_variant_choice(self):
        variants = parse_master(MASTER, BASE)
        self.assertEqual([v.height for v in variants], [1080, 720, 480])
        self.assertEqual(variants[0].url, "https://stream.example/media/hls/1080p/playlist.m3u8")
        self.assertEqual(choose_variant(variants, 1080).height, 1080)
        self.assertEqual(choose_variant(variants, 480).height, 480)
        self.assertEqual(choose_variant(variants, 600).height, 480)
        self.assertEqual(choose_variant(variants, 240).height, 480)  # en düşük varyanta düşer

    def test_media_timeline(self):
        pl = parse_media(MEDIA, "https://stream.example/media/hls/480p30/playlist.m3u8", 480)
        self.assertEqual(len(pl.segments), 3)
        self.assertAlmostEqual(pl.duration, 23.334)
        self.assertEqual(pl.program_date_time.isoformat(), "2026-11-06T19:30:09.543000+00:00")
        self.assertEqual([s.url.rsplit("/", 1)[1] for s in pl.segments_between(9.5, 12)], ["0.ts", "1.ts"])
        self.assertEqual([s.url.rsplit("/", 1)[1] for s in pl.segments_between(20, 21)], ["2.ts"])


    def test_interruptions(self):
        gap = MEDIA.replace("19:30:19.543Z", "19:30:49.543Z")  # 2. segment 30 sn geç başlıyor
        pl = parse_media(gap, "https://x/480p30/playlist.m3u8", 480)
        self.assertEqual(pl.interruptions, [(10.0, 30.0)])
        disc = MEDIA.replace("#EXTINF:3.334,", "#EXT-X-DISCONTINUITY\n#EXTINF:3.334,")
        self.assertEqual(parse_media(disc, "https://x/p.m3u8", 480).interruptions, [(20.0, 0.0)])
        self.assertEqual(parse_media(MEDIA, "https://x/p.m3u8", 480).interruptions, [])


if __name__ == "__main__":
    unittest.main()
