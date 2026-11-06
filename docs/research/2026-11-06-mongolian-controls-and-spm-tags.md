# Mongolian control characters, separated suffixes and target-language tags: research evidence

2026-11-06. Evidence note for the DECISIONS.md entry "2026-11-06: D3 follow-up -
the three open decisions, resolved by research". It holds what was read and
measured; the decisions and their consequences are in that entry, the
procedures in `docs/runbooks/03_tokenizer.md`.

## Question

D3 closed with these points open:

1. Should the `<2xx>` target-language tag be written into the text, where
   SentencePiece's default `add_dummy_prefix=True` puts a standalone `▁`
   (U+2581) in front of it, or should `add_dummy_prefix` be turned off?
2. Should `normalize` keep stripping U+200C ZWNJ and U+200D ZWJ in `mn_Mong`?
3. Which space should the traditional-script converter emit before a separated
   suffix, U+202F NNBSP or U+180E MVS, and what should `normalize` do with them?

A fourth, smaller point was researched on the way: the per-language
abbreviation sets of the Cyrillic sentence splitter.

## Sources

| Source | Address |
|---|---|
| Unicode 18 core specification, chapter 13 (read directly) | `https://www.unicode.org/versions/latest/core-spec/chapter-13/` |
| Unicode Technical Note #57 (a Technical Note, not normative) | `https://www.unicode.org/notes/tn57` |
| MNS 4932:2000, scan hosted in the Unicode document archive | `www.unicode.org/L2/topical/mongolian/MNS4932-2000(recovered).pdf` |
| W3C Mongolian Layout Requirements (a July 2025 draft note) | `https://www.w3.org/TR/mlreq/` |
| SentencePiece training options | `https://github.com/google/sentencepiece/blob/master/doc/options.md` |
| SentencePiece issues 305, 183 and 877 | `https://github.com/google/sentencepiece/issues/305`, `.../183`, `.../877` |
| CTranslate2 guide on Transformer models | `https://github.com/OpenNMT/CTranslate2/blob/master/docs/guides/transformers.md` |
| MADLAD-400 model card | `https://huggingface.co/google/madlad400-3b-mt` |
| Kazakh punctuation rules (chapter by Rabiga Syzdykova) | `https://bilim-all.kz/article/5583` |
| Mongolian regulator's list of standards | `https://www.crc.gov.mn/salbaryn-standartyn-zhagsaalt` |
| Eole 0.6.4 source, as installed in the repo environment | `eole/transforms/misc.py`, `eole/transforms/tokenize.py`, `eole/inputters/text_corpus.py` |
| Noto Sans Mongolian v3.103 release | `NotoSansMongolian-v3.103` at `github.com/notofonts/mongolian` |
| Noto Sans Mongolian 2.001 | the system build on the test machine (file dated 2020) |
| Unicode Character Database 18.0.0 | `StandardizedVariants.txt` |

## Findings

### 1. Target-language tags and `add_dummy_prefix`

- Eole 0.6.4's `PrefixTransform` (`eole/transforms/misc.py`) asserts that the
  example is already a token list (message: "PrefixTransform needs to be set
  after the Tokenizer") and prepends `side_prefix.split(" ")` to it. A
  space-separated multi-token prefix is therefore supported.
- `eole/inputters/text_corpus.py` applies a corpus's transforms in the order
  they are listed. The SentencePiece transform (`eole/transforms/tokenize.py`)
  encodes the whole string with `sp_model.encode(string, out_type=str)`.
- A tag written into the text is matched as a user-defined symbol and gets a
  standalone `▁` before it (checked: `['▁', '<2khk_Cyrl>', ...]`). With
  `transforms: [sentencepiece, prefix]` the tag is added as a token afterwards
  and no `▁` appears.
- The released SentencePiece models looked at have `add_dummy_prefix=True`:
  NLLB-200, M2M-100, mBART-50, MADLAD-400, OPUS-MT and t5-small. NLLB,
  M2M-100, mBART-50 and Marian add the language tag as an id or token after
  SentencePiece. Only MADLAD-400 puts `<2xx>` in the text, and it shows the
  same standalone `▁` as ours.
- The SentencePiece author says the dummy prefix exists so that a word is
  segmented the same wherever it appears, and that no experiments with it off
  exist (issues 305, 183 and 877).
- A toy 310-piece model: with the prefix off, a sentence-initial word was
  segmented into single letters while the same word in mid-sentence was one
  piece. The direction is certain; the size of the effect on the real model was
  not measured.
- CTranslate2 takes token lists, so the CTranslate2 wrapper can prepend the tag
  to `sp.encode(text, out_type=str)` itself.
- `check_spm` already accepts both `[symbol]` and `['▁', symbol]`.
- Not tested: running per-direction Eole corpora end to end with `eole train`.

### 2. ZWNJ and ZWJ in `mn_Mong`

- Unicode 18, chapter 13: "U+200C ZERO WIDTH NON-JOINER (ZWNJ) and U+200D ZERO
  WIDTH JOINER (ZWJ) may be used to select a particular positional form of a
  letter in isolation or to override the expected positional form within a
  word." They never distinguish words; only MVS (the specification calls its
  joined-or-separated choice purely lexical) and the free variation selectors
  (FVS) carry spelling information, and the normalizer preserves those.
- UTN #57 says everyday text does not need ZWNJ and ZWJ and recommends the
  visible U+180A MONGOLIAN NIRUGU for joining.
- U+180A..U+180F, U+200C, U+200D and U+202F all have canonical combining class
  0 and are unchanged by NFC (checked with `unicodedata`), so NFC never
  reorders them.
- MNS 4932:2000 (read from the scan) defines ZWJ and ZWNJ as positional or
  joining controls: they display a letter's initial, medial or final form alone,
  or split a word for display. When a free variation selector is also needed
  for such a standalone or separated letter, the standard puts it AFTER the
  ZWJ or ZWNJ. The Unicode standard requires the selector IMMEDIATELY after the
  base letter.
- Shaping with HarfBuzz 15.0 and Noto Sans Mongolian v3.103 accepted the selector
  in 62 of the 117 standardized variants of UCD 18.0.0 in the Unicode order
  `<base, FVS, ZWJ>`, and in 0 of 117 in the legacy order `<base, ZWJ, FVS>`.
  With the joiner stripped, `<base, FVS>`, it accepted 56 of 117. The 2020 build
  (2.001) does not label selectors as valid; there the glyph sequence differs
  between the two orders in 21 of 117 variants with ZWJ and 18 of 117 with ZWNJ.
- NFKC turns U+202F into a plain space, and SentencePiece's `nmt_nfkc` rule turns
  both ZWNJ and NNBSP into spaces (checked locally: `nmt_nfkc` gave a space for
  U+200C and U+202F and left U+200D, U+180E and U+180B alone). The trainer uses
  the `identity` rule and a test pins it.

### 3. NNBSP and MVS before a separated suffix

- Unicode 16.0 and later (re-read in the Unicode 18 text): "Prior to Unicode
  Version 16.0, U+202F NARROW NO-BREAK SPACE (NNBSP) was used to represent this
  small whitespace; ... However, its role has been taken over by U+180E MONGOLIAN
  VOWEL SEPARATOR (MVS), which not only prevents word breaking and line
  breaking, but also triggers special shaping for the following separated
  suffix."
- UTN #57 version 4 discourages NNBSP. The W3C Mongolian Layout Requirements
  draft note still documents NNBSP.
- MNS 4932:2000 (read from the scan; it predates Unicode 16) defines U+202F
  NARROW NO-BREAK SPACE as the Mongolian space for separated suffixes (section
  3.4 and item 14, with example lists). It defines MVS only as the separator of
  a word-final A or E vowel from its stem (item 12), the lexical use in the
  Unicode text.
- The font run below: Noto Sans Mongolian 2.001 shapes NNBSP plus suffix and MVS
  plus suffix differently; v3.103 produces identical glyph sequences and an
  identical render for the two. In both fonts MVS in the lexical a/e case still
  changes the glyphs, and the joiners change forms.

### 4. Abbreviations

- The official Kazakh punctuation rules (paragraphs 7-10 of the punctuation
  chapter by Rabiga Syzdykova) list the dotted abbreviations `т.б.`, `т.с.`,
  `т.т.` and `ж.` (year), `ғ.` (century), `сек.`, `мин.`, `сағ.`. Initials keep
  their dots; unit symbols such as km, m and cm take no dot.
- The Mongolian punctuation rules summary that was read uses a dot for initials
  and for abbreviations of names, and a hyphen for contractions of common words
  (`д-р` doctor, `н-р` comrade), which cannot create a sentence boundary.
- The dotted Mongolian forms for "and so on", "for example", "in other words"
  and "before the common era" are widely attested in government, health and news
  documents, but no authoritative Mongolian list was found.
- `т.с.с` and `б.з.д` were found in a Kazakh abbreviation list, not in the
  official text that was read.
- A wrong abbreviation entry is low-risk: it can only suppress a split before a
  capitalised word. Hand lists are a stopgap. The Punkt approach (Kiss and
  Strunk 2006: collocation and length statistics) would induce them from the
  real corpus. A pretrained sentence segmenter such as SaT could serve only as
  an evaluation baseline, and its Mongolian and Kazakh coverage was not
  confirmed.

## MNS 4932:2000 page references

Standard page numbers (not PDF page numbers).

| Topic | Page |
|---|---|
| The Mongolian space (section 3.4) | 3 |
| MVS (item 12) | 16 |
| Free variation selectors | 17 |
| NNBSP (item 14) | 18 |
| ZWJ and ZWNJ | 19-20 |

The scan was read as images; the translations of its items 12-16 are the
research pass's own and want a native-speaker check. Its control-character table
on page 5 has labels that do not match their code points and should not be
relied on.

## Font run

Linux, Python 3.14.10, HarfBuzz 15.0.0 through uharfbuzz 0.57.0 (fontTools
4.68.0), script `Mong`, language `mn`, left-to-right buffer. A second renderer,
Pillow-Raqm (HarfBuzz 14.5.0), gave renders that agree with the glyph runs: the
two spacers identical in v3.103 and different in 2.001. Those images and the
script that made them are not kept in the repo.

| Test | Noto Sans Mongolian 2.001 (2020 system build) | Noto Sans Mongolian v3.103 (2026) |
|---|---|---|
| NNBSP+suffix vs MVS+suffix | differ (NNBSP: narrow glyph, suffix medial; MVS: `space` glyph, suffix initial) | identical glyph sequences |
| two suffixes, NNBSP vs MVS | differ | identical |
| lexical a/e: no separator vs MVS | differ | differ |
| ZWJ/ZWNJ vs plain word | differ (form changes) | differ (form changes) |
| FVS accepted (`.valid`) in `<base,FVS,ZWJ>` / `<base,ZWJ,FVS>` / `<base,FVS>` of 117 variants | font does not label selectors; glyph sequence differs between the two orders in 21 of 117 (ZWJ) and 18 of 117 (ZWNJ) | 62 / 0 / 56 |

A script that counts glyph-name sequences also reports 117 of 117 for v3.103,
but only because the ignored joiner glyph and the selector glyph change places
in the sequence; for v3.103 the `.valid` counts are the informative measure.

## Limitations

- Fonts tested: Noto Sans Mongolian 2.001 (system, 2020) and v3.103 (2026) only,
  with HarfBuzz 15.0.0 through uharfbuzz 0.57.0 and Pillow-Raqm (HarfBuzz 14.5.0)
  on Linux and Python 3.14.10. Not tested: browsers, Windows, Mongolian Baiti,
  Pango (the attempt rendered blank). No Mongolian reader has judged the
  orthographic correctness of the renders.
- Retrieved and read: the full scan of MNS 4932:2000. The regulator's list
  (items 148-151) confirms that MNS 4333:1996, MNS 4932:2000 and MNS 5552:2005
  exist; that proves titles, not that the editions are current.
- Not retrieved: the full text of MNS 5552:2005 (it needs the national portal
  login) and its current edition (a portal notice title refers to a 2025
  revision with public comment open until 18 Aug 2024; adoption unconfirmed),
  and MNS 4333:1996.
- The combining-class and NFC check used the `unicodedata` of Python 3.14.10,
  which carries Unicode 16.0.0 data; the specification text and the variant
  list are Unicode 18.0.0.
- The `add_dummy_prefix` values of NLLB-200, M2M-100, mBART-50, OPUS-MT and
  t5-small come from the research pass over their released model files; their
  file addresses are not recorded here.
- The toy-model segmentation effect was measured on a 310-piece model only.
- No end-to-end Eole training with per-direction corpora was run.

## Appendix: reproduction harness

Three scripts rerun the font results. They build every string from hex code
points, so they hold no raw Mongolian, invisible or transliterated text; keep
them that way.

The fonts are not in the repo. Download the Noto Sans Mongolian release
`NotoSansMongolian-v3.103` from `github.com/notofonts/mongolian` and unpack it;
the file used is `NotoSansMongolian/unhinted/ttf/NotoSansMongolian-Regular.ttf`.
The 2.001 build is the system font
`/usr/share/fonts/truetype/noto/NotoSansMongolian-Regular.ttf` of the test
machine. `StandardizedVariants.txt` of UCD 18.0.0 is also needed. Treat
downloaded fonts and archives as untrusted data: unpack each into its own empty
directory, keep the scripts elsewhere, and run Python with `-I` and the file
paths as arguments, never from inside the download directory.

Environment: a Python 3.14 venv holding `uharfbuzz==0.57.0` and
`fonttools==4.68.0`.

```bash
FONT_2020=/usr/share/fonts/truetype/noto/NotoSansMongolian-Regular.ttf
FONT_V3103=<unpacked release>/NotoSansMongolian/unhinted/ttf/NotoSansMongolian-Regular.ttf
UCD=<directory>/StandardizedVariants.txt
python -I mvs_shape.py "$FONT_2020" "$FONT_V3103"
python -I fvs_valid.py "$FONT_2020" "$FONT_V3103" "$UCD"
python -I fvs_order.py "$FONT_2020" "$FONT_V3103" "$UCD"
```

Results of the run recorded above (the glyph lines of `mvs_shape.py` are omitted):

```text
FONT noto-2020(system): Version 2.001; ttfautohint (v1.8.3) -l 8 -r 50 -G 200 -x 14 -D mong -f none -a qsq -X "" | HarfBuzz 15.0.0
  -- NNBSP vs MVS before a suffix: DIFFER
  -- two suffixes, NNBSP vs MVS: DIFFER
  -- Cyrillic word, NNBSP vs MVS: DIFFER
  -- lexical a/e, no separator vs MVS: DIFFER
  -- ZWJ vs plain word: DIFFER
  -- ZWNJ inside vs plain word: DIFFER
  -- FVS+ZWJ order, Unicode vs legacy: same glyphs
FONT noto-v3.103: Version 3.103;fontc 1.0.0 | HarfBuzz 15.0.0
  -- NNBSP vs MVS before a suffix: same glyphs
  -- two suffixes, NNBSP vs MVS: same glyphs
  -- Cyrillic word, NNBSP vs MVS: same glyphs
  -- lexical a/e, no separator vs MVS: DIFFER
  -- ZWJ vs plain word: DIFFER
  -- ZWNJ inside vs plain word: DIFFER
  -- FVS+ZWJ order, Unicode vs legacy: DIFFER

noto-2020(system): FVS accepted ('.valid') in 0/117 with <base,FVS,ZWJ>; 0/117 with <base,ZWJ,FVS>; 0/117 with <base,FVS>
noto-v3.103: FVS accepted ('.valid') in 62/117 with <base,FVS,ZWJ>; 0/117 with <base,ZWJ,FVS>; 56/117 with <base,FVS>

Mongolian standardized variants in the file: 117
noto-2020(system): of 117 variants, the FVS changes the glyphs: 117; the order matters (FVS,ZWJ vs ZWJ,FVS): 21; with ZWNJ: 18
noto-v3.103: of 117 variants, the FVS changes the glyphs: 117; the order matters (FVS,ZWJ vs ZWJ,FVS): 117; with ZWNJ: 117
```

### `mvs_shape.py`

```python
"""Shape separated-suffix, vowel-separator and joiner test strings with HarfBuzz.

Usage: python -I mvs_shape.py FONT_2020 FONT_V3103
(two Noto Sans Mongolian TTF files; see the appendix of the research note)
"""

import sys

import uharfbuzz as hb
from fontTools.ttLib import TTFont

assert sys.version_info[:2] == (3, 14)
FONTS = {"noto-2020(system)": sys.argv[1], "noto-v3.103": sys.argv[2]}


def U(*hexes):  # strings are built from code points only: no raw characters here
    return "".join(chr(int(h, 16)) for h in " ".join(hexes).split())


STEM = "182E 1823 1829 182D 1823 182F"  # a word of six letters
SUFFIX_1 = "1824 1828"  # a suffix of two letters (genitive)
SUFFIX_2 = "1833 1824"  # a suffix of two letters (dative-locative)
STEM_AE = "182C 1820 1828 1820"  # the core spec's example word: no separator
STEM_AE_MVS = "182C 1820 1828 180E 1820"  # the same word with MVS before its last vowel
CYRILLIC_WORD = "041C 043E 043D 0433 043E 043B"
NNBSP, MVS, ZWJ, ZWNJ, FVS1 = "202F", "180E", "200D", "200C", "180B"

CASES = [
    ("T1 word alone", U(STEM)),
    ("T2 NNBSP + suffix", U(STEM, NNBSP, SUFFIX_1)),
    ("T3 MVS + suffix", U(STEM, MVS, SUFFIX_1)),
    ("T4a lexical a/e, no separator", U(STEM_AE)),
    ("T4b lexical a/e, with MVS", U(STEM_AE_MVS)),
    ("T5a two suffixes via NNBSP", U(STEM, NNBSP, SUFFIX_1, NNBSP, SUFFIX_2)),
    ("T5b two suffixes via MVS", U(STEM, MVS, SUFFIX_1, MVS, SUFFIX_2)),
    ("T6a Cyrillic + MVS + suffix", U(CYRILLIC_WORD, MVS, SUFFIX_1)),
    ("T6b Cyrillic + NNBSP + suffix", U(CYRILLIC_WORD, NNBSP, SUFFIX_1)),
    ("T7a word + ZWJ", U(STEM, ZWJ)),
    ("T7b ZWNJ inside the word", U("182E 1823 200C 1829 182D 1823 182F")),
    ("T7c FVS1 then ZWJ (Unicode order)", U(STEM, FVS1, ZWJ)),
    ("T7d ZWJ then FVS1 (legacy order)", U(STEM, ZWJ, FVS1)),
]


def shape(path, text):
    font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction, buf.script, buf.language = "ltr", "Mong", "mn"
    hb.shape(font, buf, {})
    names = [font.glyph_to_string(i.codepoint) for i in buf.glyph_infos]
    advances = [p.x_advance for p in buf.glyph_positions]
    return names, advances


for label, path in FONTS.items():
    version = next(n.toUnicode() for n in TTFont(path)["name"].names if n.nameID == 5)
    print("=" * 100)
    print(f"FONT {label}: {version} | HarfBuzz {hb.version_string()}")
    results = {}
    for case, text in CASES:
        names, advances = shape(path, text)
        results[case] = names
        print(f"  {case:36} n={len(names):2} adv_sum={sum(advances):5} | {' '.join(names)}")

    def verdict(a, b):
        return "DIFFER" if results[a] != results[b] else "same glyphs"

    print("  -- NNBSP vs MVS before a suffix:", verdict("T2 NNBSP + suffix", "T3 MVS + suffix"))
    print("  -- two suffixes, NNBSP vs MVS:", verdict("T5a two suffixes via NNBSP", "T5b two suffixes via MVS"))
    print("  -- Cyrillic word, NNBSP vs MVS:", verdict("T6a Cyrillic + MVS + suffix", "T6b Cyrillic + NNBSP + suffix"))
    print("  -- lexical a/e, no separator vs MVS:", verdict("T4a lexical a/e, no separator", "T4b lexical a/e, with MVS"))
    print("  -- ZWJ vs plain word:", verdict("T1 word alone", "T7a word + ZWJ"))
    print("  -- ZWNJ inside vs plain word:", verdict("T1 word alone", "T7b ZWNJ inside the word"))
    print("  -- FVS+ZWJ order, Unicode vs legacy:", verdict("T7c FVS1 then ZWJ (Unicode order)", "T7d ZWJ then FVS1 (legacy order)"))
```

### `fvs_valid.py`

```python
"""Count the standardized variants whose selector a font accepts (glyph `.valid`).

Usage: python -I fvs_valid.py FONT_2020 FONT_V3103 STANDARDIZED_VARIANTS_TXT
(STANDARDIZED_VARIANTS_TXT is StandardizedVariants.txt of UCD 18.0.0)
"""

import re
import sys

import uharfbuzz as hb

assert sys.version_info[:2] == (3, 14)
FONTS = {"noto-2020(system)": sys.argv[1], "noto-v3.103": sys.argv[2]}
ZWJ = chr(0x200D)
pairs = [
    (int(m.group(1), 16), int(m.group(2), 16))
    for m in (
        re.match(r"^(1[89][0-9A-F]{2}) (180[BCDF]);", line)
        for line in open(sys.argv[3], encoding="utf-8")
    )
    if m
]


def shape(path, text):
    font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction, buf.script, buf.language = "ltr", "Mong", "mn"
    hb.shape(font, buf, {})
    return [font.glyph_to_string(i.codepoint) for i in buf.glyph_infos]


for label, path in FONTS.items():
    n = len(pairs)
    unicode_order = legacy_order = no_joiner = 0
    for base, fvs in pairs:
        b, v = chr(base), chr(fvs)
        unicode_order += any(g.endswith(".valid") for g in shape(path, b + v + ZWJ))
        legacy_order += any(g.endswith(".valid") for g in shape(path, b + ZWJ + v))
        no_joiner += any(g.endswith(".valid") for g in shape(path, b + v))
    print(
        f"{label}: FVS accepted ('.valid') in {unicode_order}/{n} with <base,FVS,ZWJ>; "
        f"{legacy_order}/{n} with <base,ZWJ,FVS>; {no_joiner}/{n} with <base,FVS>"
    )
```

### `fvs_order.py`

```python
"""Compare the glyph sequences of the two orders of a selector and a joiner.

Usage: python -I fvs_order.py FONT_2020 FONT_V3103 STANDARDIZED_VARIANTS_TXT
The count is of glyph-name sequences that differ, nothing finer: see the note.
"""

import re
import sys

import uharfbuzz as hb

assert sys.version_info[:2] == (3, 14)
FONTS = {"noto-2020(system)": sys.argv[1], "noto-v3.103": sys.argv[2]}
ZWJ, ZWNJ = chr(0x200D), chr(0x200C)
pairs = []
for line in open(sys.argv[3], encoding="utf-8"):
    m = re.match(r"^(1[89][0-9A-F]{2}) (180[BCDF]);\s*([^;]*);\s*([^#]*)#", line)
    if m:
        pairs.append((int(m.group(1), 16), int(m.group(2), 16), m.group(4).strip()))
print("Mongolian standardized variants in the file:", len(pairs))


def shape(path, text):
    font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction, buf.script, buf.language = "ltr", "Mong", "mn"
    hb.shape(font, buf, {})
    return tuple(font.glyph_to_string(i.codepoint) for i in buf.glyph_infos)


for label, path in FONTS.items():
    total = {"with_fvs_vs_without": 0, "order_zwj": 0, "order_zwnj": 0}
    shown = 0
    for base, fvs, description in pairs:
        b, v = chr(base), chr(fvs)
        unicode_zwj, legacy_zwj = shape(path, b + v + ZWJ), shape(path, b + ZWJ + v)
        unicode_zwnj, legacy_zwnj = shape(path, b + v + ZWNJ), shape(path, b + ZWNJ + v)
        total["with_fvs_vs_without"] += unicode_zwj != shape(path, b + ZWJ)
        total["order_zwj"] += unicode_zwj != legacy_zwj
        total["order_zwnj"] += unicode_zwnj != legacy_zwnj
        if unicode_zwj != legacy_zwj and shown < 3:
            shown += 1
            print(f"   [{label}] U+{base:04X}+U+{fvs:04X} {description[:28]}: base,FVS,ZWJ={unicode_zwj}  base,ZWJ,FVS={legacy_zwj}")
    n = len(pairs)
    print(
        f"{label}: of {n} variants, the FVS changes the glyphs: {total['with_fvs_vs_without']}; "
        f"the order matters (FVS,ZWJ vs ZWJ,FVS): {total['order_zwj']}; with ZWNJ: {total['order_zwnj']}"
    )
```
