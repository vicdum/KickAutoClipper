import unittest

from kick_auto_clipper.speech import fill_gaps, monotonic, romanize
from kick_auto_clipper.subtitles import Word, build_cues

VOCAB = set("abcdefghijklmnopqrstuvwxyz'")


class SpeechTest(unittest.TestCase):
    def test_romanize_turkish(self):
        self.assertEqual(romanize("Şerefsiz!", VOCAB), "serefsiz")
        self.assertEqual(romanize("İSTİYORUM", VOCAB), "istiyorum")
        self.assertEqual(romanize("Enes'in", VOCAB), "enes'in")
        self.assertEqual(romanize("11'e", VOCAB), "'e")
        self.assertEqual(romanize("3.000", VOCAB), "")

    def test_fill_gaps_interpolates_unaligned_words(self):
        aligned = [(0.0, 0.5, 0.9), None, None, (2.0, 2.5, 0.9)]
        whisper = [(0.0, 0.5), (0.6, 0.9), (0.9, 1.5), (2.0, 2.5)]
        out = fill_gaps(aligned, whisper)
        self.assertEqual(out[0], (0.0, 0.5))
        self.assertEqual(out[3], (2.0, 2.5))
        # Aradaki iki kelime 0.5-2.0 aralığını Whisper sürelerine (0.3, 0.6) orantılı paylaşır.
        self.assertAlmostEqual(out[1][0], 0.5)
        self.assertAlmostEqual(out[1][1], 1.0)
        self.assertAlmostEqual(out[2][1], 2.0)

    def test_fill_gaps_at_edges(self):
        out = fill_gaps([None, (1.0, 1.5, 0.9), None], [(0.2, 0.8), (1.0, 1.5), (1.6, 2.0)])
        self.assertEqual(len(out), 3)
        self.assertLessEqual(out[0][1], 1.0)
        self.assertGreaterEqual(out[2][0], 1.5)

    def test_monotonic(self):
        self.assertEqual(monotonic([(6.4, 6.6), (6.3, 7.0), (6.8, 6.9)], 5.0), [(6.4, 6.6), (6.6, 7.0), (7.0, 7.0)])
        self.assertEqual(monotonic([(1.0, 2.0)], 3.0), [(3.0, 3.0)])

    def test_cues_break_at_segment_boundary_without_punctuation(self):
        ws = [Word(0, 0.3, "ananın"), Word(0.3, 0.6, "amcığı", sentence_end=True), Word(0.6, 0.9, "Şurada")]
        self.assertEqual([c.text for c in build_cues(ws)], ["ananın amcığı", "Şurada"])


if __name__ == "__main__":
    unittest.main()
