import random
import unittest

from kick_auto_clipper.detect import DetectParams, detect, effective_baseline, rolling_median, significant_rate


def msg(t, content="KEKW", user=None):
    user = user or f"u{random.random()}"
    return {"t": t, "content": content, "user": user, "user_id": user}


class DetectTest(unittest.TestCase):
    def synthetic(self, spikes, duration=1800, content="KEKW", base_per_min=20):
        """Dakikada `base_per_min` mesajlık taban + verilen aralıklarda dakikada ~200 mesaj."""
        rng = random.Random(0)
        msgs = [msg(rng.uniform(0, duration)) for _ in range(duration * base_per_min // 60)]
        for s, e in spikes:
            msgs += [msg(rng.uniform(s, e), content) for _ in range((e - s) * 200 // 60)]
        return sorted(msgs, key=lambda m: m["t"])

    def test_finds_spikes_with_padding(self):
        clips = detect(self.synthetic([(600, 630), (1200, 1220)]), 1800, DetectParams(max_duration=180)).clips
        self.assertEqual(len(clips), 2)
        for c, (s, e) in zip(clips, [(600, 630), (1200, 1220)]):
            # Ortalanmış pencere nedeniyle sınırlar birkaç saniye oynayabilir.
            self.assertAlmostEqual(c.spike_start, s, delta=15)
            self.assertAlmostEqual(c.spike_end, e, delta=15)
            self.assertEqual(c.start, c.spike_start - 15)
            self.assertEqual(c.end, c.spike_end + 15)
            self.assertGreater(c.score, 2)

    def test_quiet_stream_has_no_clips(self):
        self.assertEqual(detect(self.synthetic([]), 1800).clips, [])

    def test_intro_and_outro_are_rejected(self):
        res = detect(self.synthetic([(20, 50), (900, 920), (1760, 1790)]), 1800)
        self.assertEqual(len(res.clips), 1)
        self.assertEqual(sorted(r.reason for r in res.rejected), ["açılış", "kapanış"])

    def test_noise_messages_do_not_create_spikes(self):
        for content in ("f5", "selam", "bb", "!puan"):
            with self.subTest(content=content):
                self.assertEqual(detect(self.synthetic([(900, 930)], content=content), 1800).clips, [])

    def test_spike_after_interruption_is_rejected(self):
        res = detect(self.synthetic([(900, 920)]), 1800, interruptions=[(880.0, 30.0)])
        self.assertEqual(res.clips, [])
        self.assertEqual(res.rejected[0].reason, "yayın kesintisi")

    def test_single_user_spam_is_ignored(self):
        msgs = self.synthetic([])
        msgs += [msg(900 + i * 0.2, user="spammer") for i in range(150)]
        self.assertEqual(detect(sorted(msgs, key=lambda m: m["t"]), 1800).clips, [])

    def test_long_spike_is_trimmed_to_max_duration(self):
        clips = detect(self.synthetic([(600, 700)]), 1800, DetectParams(max_duration=60)).clips
        self.assertEqual(len(clips), 1)
        self.assertLessEqual(clips[0].duration, 60)
        self.assertGreaterEqual(clips[0].start, clips[0].spike_start - 15)

    def test_overlapping_clips_merge_within_limit(self):
        clips = detect(self.synthetic([(600, 610), (630, 640)]), 1800).clips
        self.assertEqual(len(clips), 1)
        self.assertLessEqual(clips[0].duration, 60)

    def test_overlapping_clips_do_not_overlap_when_too_long(self):
        clips = detect(self.synthetic([(600, 620), (668, 688)]), 1800).clips
        self.assertEqual(len(clips), 2)
        self.assertLessEqual(clips[0].end, clips[1].start)

    def test_max_clips_keeps_highest_scores(self):
        clips = detect(self.synthetic([(300, 320), (900, 920), (1500, 1520)]), 1800, DetectParams(max_clips=2)).clips
        self.assertEqual(len(clips), 2)
        self.assertLessEqual(clips[0].start, clips[1].start)

    def test_quiet_chat_fluctuation_is_not_a_spike(self):
        # Dakikada ~4 mesajlık sessiz sohbette dakikada ~12'ye çıkan kısa dalgalanma tesadüf sayılmalı.
        msgs = self.synthetic([], base_per_min=4)
        rng = random.Random(1)
        msgs += [msg(900 + rng.uniform(0, 30)) for _ in range(4)]
        self.assertEqual(detect(sorted(msgs, key=lambda m: m["t"]), 1800).clips, [])
        # Aynı sohbette güçlü bir sıçrama yine bulunmalı.
        self.assertEqual(len(detect(self.synthetic([(900, 930)], base_per_min=4), 1800).clips), 1)

    def test_significant_rate_is_stricter_for_quiet_chats(self):
        self.assertGreater(significant_rate(5, 30, 1e-4) / 5, 4)  # sessiz: 4 kattan fazla
        self.assertLess(significant_rate(200, 30, 1e-4) / 200, 2)  # kalabalık: ratio=2 baskın

    def test_few_users_spike_is_rejected(self):
        # Dakikada 1 sabit zamanlı arka plan mesajı (patlama penceresine düşmez) + 3 kişinin patlaması.
        msgs = [msg(60 * k + 45) for k in range(30)]
        msgs += [msg(900 + i * 0.3, user=f"u{i % 3}") for i in range(100)]
        res = detect(sorted(msgs, key=lambda m: m["t"]), 1800, DetectParams(user_cooldown=0))
        self.assertEqual(res.clips, [])
        self.assertTrue(res.rejected[0].reason.startswith("az kullanıcı"))

    def test_giveaway_keyword_flood_is_rejected(self):
        msgs = self.synthetic([(900, 930)], content="palavip")
        msgs.append(msg(600, "ANAHTAR KELİME VERİLECEK CHATE ONU YAZAN KATILMIŞ OLACAK", user="mod"))
        res = detect(sorted(msgs, key=lambda m: m["t"]), 1800)
        self.assertEqual(res.clips, [])
        self.assertEqual(res.rejected[0].reason, "çekiliş anahtar kelimesi (palavip)")

    def test_spike_with_giveaway_mentions_is_rejected(self):
        msgs = self.synthetic([(900, 930)])
        msgs += [msg(850 + i, "çekilişi ben kazanırsam ne yapacağım", user=f"v{i}") for i in range(2)]
        msgs.append(msg(890, "🎉 BÜYÜK ÇEKİLİŞ BAŞLIYOR! 🎉", user="BotRix"))  # bot duyurusu da kanıttır
        res = detect(sorted(msgs, key=lambda m: m["t"]), 1800)
        self.assertEqual(res.clips, [])
        self.assertEqual(res.rejected[0].reason, "çekiliş")

    def test_bot_command_lists_are_not_giveaway_mentions(self):
        # "!8ball !ai !cekilis ..." gibi komut listeleri ve "!cekilis" komutu çekiliş kanıtı sayılmaz.
        msgs = self.synthetic([(900, 930)])
        msgs += [msg(850 + i, "komutlar: !ai !aktif !benim !bot !cekilis", user="BotRix") for i in range(3)]
        msgs += [msg(860 + i, "!cekilis", user=f"c{i}") for i in range(3)]
        self.assertEqual(len(detect(sorted(msgs, key=lambda m: m["t"]), 1800).clips), 1)

    def test_giveaway_session_rejects_reactions_around_entry_floods(self):
        msgs = self.synthetic([(900, 930), (1100, 1130)])
        msgs = [m for m in msgs if not (900 <= m["t"] < 930)]
        msgs += [msg(900 + i * 0.2, "palavip") for i in range(150)]  # katılım akını
        msgs.append(msg(500, "ANAHTAR KELİME VERİLECEK", user="mod"))
        res = detect(sorted(msgs, key=lambda m: m["t"]), 1800)
        reasons = sorted(r.reason for r in res.rejected)
        self.assertEqual(res.clips, [])
        self.assertIn("çekiliş oturumu", reasons)  # 1100'deki tepki, akından 200 sn sonra

    def test_keyword_flood_without_giveaway_mention_is_kept(self):
        # Çekiliş anılmadıysa toplu tek kelime (ör. "şike") gerçek tepki sayılır.
        self.assertEqual(len(detect(self.synthetic([(900, 930)], content="şike"), 1800).clips), 1)

    def test_rolling_median(self):
        values = [1.0] * 50 + [100.0] * 5 + [1.0] * 50
        self.assertEqual(rolling_median(values, 21, step=1)[52], 1.0)

    def test_effective_baseline_limits_local_drift(self):
        # Yerel taban genelin 4 katına çıksa da etkin taban 2 katta kalır.
        self.assertEqual(effective_baseline([40.0, 10.0, 0.0], 10.0), [20.0, 10.0, 10.0])


if __name__ == "__main__":
    unittest.main()
