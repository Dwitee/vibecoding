import re
import nltk
from nltk.corpus import wordnet as wn
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# ── one-off downloads (comment out after first run) ───────────────────
# nltk.download("wordnet"); nltk.download("omw-1.4")
# python -m spacy download en_core_web_sm
# ---------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# ───────────── 1.  L E X I C O N S  ───────────────────────────────────
_SEED_VERBS = [
    "sound", "make_noise", "bang", "clap", "crash", "creak", "clang",
    "shatter", "smash", "slam", "clatter", "click", "knock", "ring",
]
_SEED_NOUNS = ["noise", "sound"]

def _hyponyms_of(*lemmas, pos):
    out, seen, stack = set(), set(), [syn for l in lemmas for syn in wn.synsets(l, pos=pos)]
    while stack:
        syn = stack.pop()
        if syn in seen:
            continue
        seen.add(syn)
        out.update(l.name().replace("_", " ") for l in syn.lemmas())
        stack.extend(syn.hyponyms())
    return out

def _words_whose_gloss_mentions(needle, pos):
    needle = needle.lower()
    return {
        l.name().replace("_", " ")
        for syn in wn.all_synsets(pos=pos)
        if needle in syn.definition().lower()
        for l in syn.lemmas()
    }

# -------- verbs --------
SOUND_VERBS = (
    _hyponyms_of(*_SEED_VERBS, pos="v")
    | _words_whose_gloss_mentions("sound", pos="v")
    | _words_whose_gloss_mentions("noise", pos="v")
)

# -------- nouns --------
SOUND_NOUNS = (
    _hyponyms_of(*_SEED_NOUNS, pos="n")
    | _words_whose_gloss_mentions("sound", pos="n")
    | _words_whose_gloss_mentions("noise", pos="n")
    | _words_whose_gloss_mentions("clap",  pos="n")     # ▶ NEW  ◀
)

# also pull in nouns derivationally-related to any sound verb
for v in SOUND_VERBS:
    for syn in wn.synsets(v, pos="v"):
        for v_lemma in syn.lemmas():
            for rel in v_lemma.derivationally_related_forms():
                if rel.synset().pos() == "n":
                    SOUND_NOUNS.add(rel.name().replace("_", " "))

# ───────────── 2.  G E R U N D  helper ────────────────────────────────
_CVC = re.compile(r"^[^aeiouy]*[aeiou][^aeiouy]$")

def to_gerund(base):
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        return base[:-1] + "ing"
    if base.endswith("ic"):
        return base + "king"
    if _CVC.match(base) and len(base) <= 3:
        return base + base[-1] + "ing"
    return base + "ing"

# ───────────── 3-A.  raw extraction ───────────────────────────────────
def _raw_sound_keywords(text):
    doc, found = nlp(text), set()

    for tok in doc:
        # (A) verb patterns
        if tok.pos_ == "VERB" and tok.lemma_ in SOUND_VERBS:
            gerund = tok.text if tok.tag_ == "VBG" else to_gerund(tok.lemma_)
            dobj = next((c for c in tok.children if c.dep_ in ("dobj", "obj")), None)
            if dobj:
                found.add(f"{dobj.text.lower()} {gerund}")
            else:
                subj = next((c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                if subj:
                    found.add(f"{subj.text.lower()} {gerund}")

        # (B) stand-alone sound nouns
        if tok.pos_ == "NOUN" and tok.lemma_ in SOUND_NOUNS:
            found.add(tok.lemma_.lower())

    return sorted(found)

# ───────────── 3-B.  clean-up ─────────────────────────────────────────
_PRONOUNS = {
    "i", "me", "you", "he", "him", "she", "her", "it",
    "we", "us", "they", "them", "someone", "something",
}

def _cleanup_keywords(raw):
    # 1. drop pronoun / stop-word leaders
    cand = [kw for kw in raw if kw.split()[0] not in _PRONOUNS and kw.split()[0] not in STOP_WORDS]
    # 2. keep-longest-only dedupe
    cand.sort(key=len, reverse=True)
    keep = []
    for kw in cand:
        if not any(kw in longer for longer in keep):
            keep.append(kw)
    return sorted(keep)

# ─────────────  P U B L I C   A P I  ──────────────────────────────────
def extract_sound_keywords(text: str):
    """Single call → tidy, non-redundant sound keywords."""
    return _cleanup_keywords(_raw_sound_keywords(text))

# ─────────────  demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    demo = (
        "Car horns blared in stop-and-go traffic while motorbikes revved impatiently between lanes. A vendor's bell chimed as he pushed a cart, and distant sirens wailed, weaving through the chaos. Pigeons flapped noisily when a taxi door slammed shut."
    )
    print(extract_sound_keywords(demo))
    
    
    # test
    # curl -X POST http://localhost:5000/generateKeywordsFromText -H "Content-Type: application/json" -d '{"text": "Car horns blared in stop-and-go traffic while motorbikes revved impatiently between lanes. A vendors bell chimed as he pushed a cart, and distant sirens wailed, weaving through the chaos. Pigeons flapped noisily when a taxi door slammed shut."}'