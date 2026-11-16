import unittest

from kick_auto_clipper.subtitles import (
    _HALLUCINATIONS, Word, _latin, build_cues, looks_english, shift, to_ass, to_srt, upper,
)


def words(*items):
    return [Word(s, e, t) for s, e, t in items]


class SubtitlesTest(unittest.TestCase):
    def test_cues_split_on_word_limit_pause_and_sentence_end(self):
        ws = words(
            (0.0, 0.3, "bir"), (0.3, 0.6, "iki"), (0.6, 0.9, "üç"), (0.9, 1.2, "dört"),  # 3 kelime sınırı
            (3.0, 3.3, "uzun"),  # 0.6 sn'den uzun duraklama
            (3.3, 3.6, "bitti."), (3.6, 3.9, "yeni"),  # cümle sonu
        )
        self.assertEqual([c.text for c in build_cues(ws)], ["bir iki üç", "dört", "uzun bitti.", "yeni"])

    def test_cues_respect_char_limit(self):
        ws = words((0, 1, "çokuzunbirkelime"), (1, 2, "dahauzunbirkelime"))
        self.assertEqual(len(build_cues(ws, max_chars=22)), 2)

    def test_short_gaps_are_closed(self):
        cues = build_cues(words((0, 1, "a."), (1.2, 2, "b")))
        self.assertEqual(cues[0].end, 1.2)

    def test_srt(self):
        srt = to_srt(build_cues(words((61.5, 62.25, "merhaba"))))
        self.assertEqual(srt, "1\n00:01:01,500 --> 00:01:02,250\nmerhaba\n\n")

    def test_ass_highlights_each_word(self):
        ass = to_ass(build_cues(words((0, 0.5, "hadi"), (0.5, 1.0, "bakalım"))), 1080, 1920, language="tr")
        events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
        self.assertEqual(len(events), 2)
        self.assertIn("{\\c&H00E5FF&}HADİ{\\c&HFFFFFF&} BAKALIM", events[0])
        self.assertIn("HADİ {\\c&H00E5FF&}BAKALIM{\\c&HFFFFFF&}", events[1])
        self.assertIn("PlayResX: 1080", ass)

    def test_ass_escapes_override_braces(self):
        ass = to_ass(build_cues(words((0, 1, "{\\b1}x"))), 480, 854)
        self.assertNotIn("{\\b1}", ass)

    def test_turkish_upper(self):
        self.assertEqual(upper("şimdi ılık", "tr"), "ŞİMDİ ILIK")
        self.assertEqual(upper("this", "en"), "THIS")

    def test_shift(self):
        cue = shift(build_cues(words((1, 2, "a"))), 0.5)[0]
        self.assertEqual((cue.start, cue.end, cue.words[0].start), (1.5, 2.5, 1.5))

    def test_language_helpers(self):
        self.assertTrue(_latin("Şuraya dokun, abi!"))
        self.assertTrue(_latin("Let's get real. Café"))
        self.assertFalse(_latin("Редактор субтитров А.Семкин"))
        self.assertTrue(looks_english("I just had the longest meeting."))
        self.assertTrue(looks_english("Oh, man, am I a woman?"))
        self.assertFalse(looks_english("Sütü karıştır, Pam."))
        self.assertFalse(looks_english("Bunlar 10 yıl önce abi"))
        self.assertFalse(looks_english("Kelly!"))  # yaygın İngilizce kelime yok -> yayın dili

    def test_ass_uppercase_follows_word_language(self):
        cues = build_cues([Word(0, 0.5, "this", lang="en"), Word(0.5, 1.0, "bitti", lang="tr")])
        ass = to_ass(cues, 1080, 1920, language="tr")
        self.assertIn("THIS", ass)  # İngilizce: i -> I
        self.assertIn("BİTTİ", ass)  # Türkçe: i -> İ

    def test_hallucination_filter(self):
        for t in ("Altyazı M.K.", "Abone olmayı unutmayın", "İzlediğiniz için teşekkürler",
                  "Редактор субтитров А.Семкин Корректор А.Егорова"):
            self.assertTrue(_HALLUCINATIONS.search(t), t)
        self.assertIsNone(_HALLUCINATIONS.search("Bedava verdik mi?"))
        # Ses etiketleri yalnızca segmentin tamamıysa elenir; cümle içinde geçen kelime kalır.
        for t in (" Müzik", "[Müzik]", "(Alkış)", "Gülüşmeler.", "♪ Music ♪"):
            self.assertTrue(_HALLUCINATIONS.search(t.strip()), t)
        self.assertIsNone(_HALLUCINATIONS.search("Bu müzik çok güzel"))


if __name__ == "__main__":
    unittest.main()
