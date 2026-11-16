import random
import unittest

from kick_auto_clipper.detect import DetectParams, detect, rolling_median


class DetectTest(unittest.TestCase):
    def synthetic(self, spikes: list[tuple[int, int]], duration: int = 1800) -> list[float]:
        """Dakikada ~20 mesajlık taban + verilen aralıklarda dakikada ~200 mesaj."""
        rng = random.Random(0)
        times = [rng.uniform(0, duration) for _ in range(duration * 20 // 60)]
        for s, e in spikes:
            times += [rng.uniform(s, e) for _ in range((e - s) * 200 // 60)]
        return sorted(times)

    def test_finds_spikes_with_padding(self):
        clips = detect(self.synthetic([(600, 630), (1200, 1220)]), 1800)
        self.assertEqual(len(clips), 2)
        for c, (s, e) in zip(clips, [(600, 630), (1200, 1220)]):
            # Ortalanmış pencere nedeniyle sınırlar birkaç saniye oynayabilir.
            self.assertAlmostEqual(c.spike_start, s, delta=15)
            self.assertAlmostEqual(c.spike_end, e, delta=15)
            self.assertEqual(c.start, max(0, c.spike_start - 15))
            self.assertEqual(c.end, c.spike_end + 15)
            self.assertGreater(c.score, 2)

    def test_quiet_stream_has_no_clips(self):
        self.assertEqual(detect(self.synthetic([]), 1800), [])

    def test_overlapping_clips_merge(self):
        clips = detect(self.synthetic([(600, 620), (650, 670)]), 1800)
        self.assertEqual(len(clips), 1)

    def test_max_clips_keeps_highest_scores(self):
        times = self.synthetic([(300, 320), (900, 920), (1500, 1520)])
        clips = detect(times, 1800, DetectParams(max_clips=2))
        self.assertEqual(len(clips), 2)
        self.assertLessEqual(clips[0].start, clips[1].start)

    def test_rolling_median(self):
        values = [1.0] * 50 + [100.0] * 5 + [1.0] * 50
        self.assertEqual(rolling_median(values, 21, step=1)[52], 1.0)


if __name__ == "__main__":
    unittest.main()
