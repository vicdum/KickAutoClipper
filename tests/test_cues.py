import unittest

from kick_auto_clipper.subtitles import (
    Cue, Word, build_cues, fix_long_words, natural_duration, no_overlap, shift_cue, text_similarity,
)


def cue(start, end, *texts):
    step = (end - start) / len(texts)
    return Cue(start, end, [Word(start + i * step, start + (i + 1) * step, t) for i, t in enumerate(texts)])


class CueTest(unittest.TestCase):
    def test_cues_break_at_segment_boundary_without_punctuation(self):
        ws = [Word(0, 0.3, "ananın"), Word(0.3, 0.6, "amcığı", sentence_end=True), Word(0.6, 0.9, "Şurada")]
        self.assertEqual([c.text for c in build_cues(ws)], ["ananın amcığı", "Şurada"])

    def test_shift_cue_moves_words_and_clamps_at_zero(self):
        c = shift_cue(cue(1.0, 2.0, "a", "b"), -1.5)
        self.assertEqual((c.start, c.end), (0.0, 0.5))
        self.assertEqual(c.words[1].end, 0.5)

    def test_no_overlap_trims_previous_cue(self):
        out = no_overlap([cue(0.0, 2.0, "a", "b"), cue(1.5, 3.0, "c")])
        self.assertEqual(out[0].end, 1.5)
        self.assertTrue(all(w.end <= 1.5 for w in out[0].words))
        self.assertEqual(out[1].start, 1.5)

    def test_no_overlap_sorts_after_shift(self):
        out = no_overlap([cue(3.0, 4.0, "b"), cue(1.0, 2.0, "a")])
        self.assertEqual([c.text for c in out], ["a", "b"])

    def test_fix_long_words(self):
        ws = [
            Word(0.0, 3.0, "Bir"),  # cümlenin ilki: başlangıç korunur
            Word(3.0, 7.0, "dakika", sentence_end=True),  # cümle içi: bitiş korunur
            Word(7.0, 7.4, "abi"),  # normal uzunluk: dokunulmaz
        ]
        out = fix_long_words(ws)
        self.assertEqual((out[0].start, out[0].end), (0.0, round(natural_duration("Bir"), 3)))
        self.assertEqual((out[1].start, out[1].end), (round(7.0 - natural_duration("dakika"), 3), 7.0))
        self.assertEqual((out[2].start, out[2].end), (7.0, 7.4))
        # Kırpılan aralıkta oluşan boşluk satırı böler: "dakika" ayrı satırda, söylendiği anda başlar.
        self.assertEqual([c.text for c in build_cues(out)], ["Bir", "dakika", "abi"])

    def test_text_similarity_ignores_case_accents_and_spaces(self):
        self.assertEqual(text_similarity("Ananı sikeyim.", "ananı sikeyim"), 1.0)
        self.assertEqual(text_similarity("Şuradan gel", "SURADAN GEL"), 1.0)
        # İlgisiz cümleler ortak harflerden ~0,4 alır; düzeltme eşikleri (zayıf < 0,5, kabul >= 0,6) bunun üstünde.
        self.assertLess(text_similarity("Bedava verdik mi?", "Teşekkür ederim."), 0.5)
        self.assertEqual(text_similarity("abc", ""), 0.0)


if __name__ == "__main__":
    unittest.main()
