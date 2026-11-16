import unittest

from kick_auto_clipper.noise import COMMAND, FAREWELL, GREETING, TECHNICAL, classify


class NoiseTest(unittest.TestCase):
    def check(self, kind, *messages, user="izleyici"):
        for m in messages:
            with self.subTest(m=m):
                self.assertEqual(classify(m, user), kind)

    def test_greeting(self):
        self.check(GREETING, "sa", "SA", "selam beyler", "as hg", "Hoş geldin hocam", "selamün aleyküm")

    def test_farewell(self):
        self.check(FAREWELL, "bb", "görüşürüz hocam", "iyi geceler", "kapama aq", "[emote:5748106:collectiblesbye]")

    def test_technical(self):
        self.check(TECHNICAL, "f5", "F5 F5", "hoca yayın gitti", "geri geldi", "drop", "dondu", "ses yok")

    def test_command(self):
        self.check(COMMAND, "!izleme top", "!clip 30")
        self.check(COMMAND, "@birisi 3. seviyeye yükseldin.", user="BotRix")

    def test_content_is_not_noise(self):
        self.check(
            None,
            "[emote:37226:KEKW]",
            "faul",
            "arkada 3. dünya savaşı dönüyor",  # dönüyor != donuyor
            "hoca suriyeden döndü",
            "[emote:4148074:HYPERCLAP]",
            "[emote:1:gachiGASM]",  # 'hi' içeren emote selam sayılmamalı
            "hocam bugün bilardoda selam verdi mi kimseye bilmiyorum ama çok iyi oynadı",
        )


if __name__ == "__main__":
    unittest.main()
