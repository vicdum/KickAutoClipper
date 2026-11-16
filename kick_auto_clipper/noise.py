"""Sohbet mesajlarını içerik açısından sınıflandırır.

Viral anlarla ilgisi olmayan ama toplu yazıldığı için sıçrama üreten mesajlar:
- greeting:  yayın açılışındaki selamlaşma (selam, sa, hg, merhaba...)
- farewell:  kapanıştaki vedalaşma (bb, görüşürüz, iyi geceler, kapatma...)
- technical: yayın kesintisi tepkileri (drop, dondu, f5, ses yok, geri geldi...)
- command:   bot komutları (!puan, !zaman...) ve bot hesapları

Bu mesajlar sıçrama hesabına katılmaz; bir sıçramanın büyük kısmı bunlardan
oluşuyorsa sıçrama reddedilir.
"""

from __future__ import annotations

import re

GREETING = "greeting"
FAREWELL = "farewell"
TECHNICAL = "technical"
COMMAND = "command"
NOISE_KINDS = (GREETING, FAREWELL, TECHNICAL, COMMAND)

# Normalize edilmiş (küçük harf, Türkçe karakterler ASCII'ye çevrilmiş) kelime/ifadeler.
_KEYWORDS: dict[str, list[str]] = {
    GREETING: [
        "selam", "selamlar", "slm", "slmlar", "sa", "sea", "selamun aleykum", "selamin aleykum",
        "as", "aleykum selam", "aleykumselam", "merhaba", "merhabalar", "mrb", "mrv", "hi", "hello",
        "hey", "heyy", "hosgeldin", "hosgeldiniz", "hos geldin", "hg", "gunaydin", "naber",
        "iyi aksamlar", "iyi yayinlar", "yayin acildi", "hayirli yayinlar",
    ],
    FAREWELL: [
        "bb", "bbb", "bye", "byee", "bay bay", "baybay", "bay", "gorusuruz", "gorusmek uzere",
        "grsz", "grsrz", "iyi geceler", "iyi uykular", "ig", "kib", "kendine iyi bak", "hoscakal",
        "hosca kal", "kapatma", "kapama", "kapatmayin", "kapatiyor", "yayin bitti", "yayin bitiyor",
        "yayin icin tesekkurler", "emegine saglik", "eline saglik", "yarin gorusuruz",
    ],
    TECHNICAL: [
        "drop", "dropp", "dropped", "ekran dondu", "goruntu dondu", "dondu mu", "kasiyor", "kasma",
        "lag", "lagg", "laggy", "f5", "f8", "refresh", "buffer", "yukleniyor", "yayin gitti",
        "yayin koptu", "yayin dondu", "yayin kesildi", "yayin nerde", "yayin nerede", "yayin yok",
        "koptu", "kesildi", "kesiliyor", "ses yok", "ses gitti", "ses gelmiyor", "goruntu yok",
        "goruntu gitti", "siyah ekran", "gitti mi", "geri geldi", "yayin geldi", "geldi mi",
        "baglanti", "internet gitti", "net gitti", "kick coktu",
    ],
}

# Emote adında geçtiğinde sınıfı belirleyen parçalar (ör. collectiblesBye, peepoWave).
_EMOTE_HINTS = {FAREWELL: ("bye", "wave"), GREETING: ("hello",)}

# Türkçe karakterler ASCII'ye çevrilince anlamı karışan kelimeler (dondu/döndü,
# donuyor/dönüyor): yalnızca mesajın tamamı bu kelimeden oluşuyorsa sayılır.
_EXACT_ONLY: dict[str, set[str]] = {TECHNICAL: {"dondu", "donuyor", "donma", "donmus", "dondu mu"}}

_BOT_USERS ={"botrix", "kickbot", "fossabot", "nightbot", "streamelements", "moobot"}
_MAX_TOKENS = 6  # uzun mesajlar içerik taşır; yalnızca kısa mesajlar gürültü sayılır

_TR = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosucgiosuaiu")
_EMOTE = re.compile(r"\[emote:\d+:([^\]]+)\]")
_PHRASES = {
    kind: [tuple(p.split()) for p in sorted(words, key=lambda w: -len(w.split()))]
    for kind, words in _KEYWORDS.items()
}


def normalize(text: str) -> str:
    text = text.replace("İ", "i").replace("I", "ı").lower().translate(_TR)
    text = text.replace("̇", "")  # birleşik nokta (i̇)
    return re.sub(r"[^\w\s]", " ", text)


def _contains(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    n = len(phrase)
    return any(tuple(tokens[i : i + n]) == phrase for i in range(len(tokens) - n + 1))


def classify(content: str, user: str | None = None) -> str | None:
    """Mesaj gürültüyse türünü, değilse None döner."""
    raw = content.strip()
    if raw.startswith("!") or (user and (user.lower() in _BOT_USERS or user.lower().endswith("bot"))):
        return COMMAND

    emotes = [e.lower() for e in _EMOTE.findall(raw)]
    tokens = normalize(_EMOTE.sub(" ", raw)).split()

    for kind, words in _EXACT_ONLY.items():
        # "dondu" ya da "dondu dondu dondu" gibi tekrarlar da sayılır.
        if " ".join(tokens) in words or (tokens and len(set(tokens)) == 1 and tokens[0] in words):
            return kind
    if tokens and len(tokens) <= _MAX_TOKENS:
        for kind in (TECHNICAL, FAREWELL, GREETING):
            if any(_contains(tokens, p) for p in _PHRASES[kind]):
                return kind
    if not tokens and emotes:
        for kind, hints in _EMOTE_HINTS.items():
            if all(any(h in e for h in hints) for e in emotes):
                return kind
    return None
