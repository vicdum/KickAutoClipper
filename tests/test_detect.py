import random
import unittest

from kick_auto_clipper.detect import DetectParams, detect, effective_baseline, rolling_median


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

    def test_rolling_median(self):
        values = [1.0] * 50 + [100.0] * 5 + [1.0] * 50
        self.assertEqual(rolling_median(values, 21, step=1)[52], 1.0)

    def test_effective_baseline_limits_local_drift(self):
        # Yerel taban genelin 4 katına çıksa da etkin taban 2 katta kalır.
        self.assertEqual(effective_baseline([40.0, 10.0, 0.0], 10.0), [20.0, 10.0, 10.0])


if __name__ == "__main__":
    unittest.main()
