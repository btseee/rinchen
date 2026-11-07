# Wiktionary Khalkha Cyrillic to traditional-script reference

Generated 2026-11-07 07:16Z. Wikitext fetched 2026-11-07 07:11Z from the English Wiktionary MediaWiki API (action=query, prop=revisions).
Scripts: `fetch_titles.py`, `fetch_pages.py`, `parse.py`, `check_html.py`, `make_report.py`, `validate.py` (all in this directory; CPython 3.14, run with `-I`).

## Counts

- Category members scanned: {'Category:Mongolian_lemmas': 3970, 'Category:Mongolian_nouns': 2618, 'Category:Mongolian_verbs': 474, 'Category:Mongolian_adjectives': 362}
- Candidate Cyrillic titles (lowercase, 2-12 letters, no space or hyphen): 2689
- Titles fetched (wikitext plus revision id): 2689 (missing pages: 0)
- Entries kept: **2029** (aim was 300-1000; every kept row passed all rules, so no arbitrary cut was made, see caveats)
- Kept rows with FVS/MVS/NNBSP in the script (`has_fvs_mvs_nnbsp=yes`): 350
- Kept rows with part of speech `?`: 0

Rejections by reason (first failing rule, applied in this order):

| reason | pages |
|---|---|
| restricted_label | 45 |
| restricted_label_other_lines | 17 |
| verification_or_cleanup_requested | 8 |
| variant_cyrillic_mismatch | 1 |
| no_script_data | 476 |
| mongolia_vs_china_spelling_differs | 1 |
| multiple_distinct_scripts | 88 |
| invalid_char:zwnj_or_zwj | 10 |
| invalid_char:u1880_and_above | 1 |
| invalid_char:u1843_to_u187f | 0 |
| invalid_char:ascii_space | 4 |
| invalid_char:other_non_mongolian | 9 |
| total rejected | 660 |

Reason notes:

- `restricted_label`: a label-bearing template (`lb`, `label`, `lbl`, `q`, `qualifier`, ...) on a definition line of the Mongolian section names Inner Mongolia, China, Chakhar, Southern Mongolia, Classical, dated, obsolete or archaic (also Inner Mongolian dialect names, Oirat, Buryat, Kalmyk, Tuvan), or a definition line mentions Inner Mongolia, Chakhar or Southern Mongolia.
- `restricted_label_other_lines`: the same hit, but only outside definition lines (for example a synonym list entry labelled dated, or a usage note). Rejected anyway (conservative), reported separately.
- `verification_or_cleanup_requested`: the section carries `rfv`, `rfc` or `rfd` templates.
- `no_script_data`: the Mongolian section has no script string in any headword or variant template.
- `multiple_distinct_scripts`: more than one distinct string came out of the section (52 from `mn-variant` positional arguments that disagree, 36 where headword templates of different parts of speech, or a headword template and an `mn-variant`, give different strings).
- `mongolia_vs_china_spelling_differs`: `{{mn-variant|mn=...|cn=...}}` gives different Mongolia and China spellings (the China form is excluded by rule, and two strings are present).
- `invalid_char:*`: the single string contains a code point outside U+1820..U+1842, U+180B..U+180F, U+202F. Includes comma-separated alternative spellings inside one argument (`other_non_mongolian`), multiword forms with an ASCII space or wikilink brackets, and ZWJ/ZWNJ.

## Templates the data came from

Only structured templates inside the `==Mongolian==` section were read: `{{mn-variant|<script>...}}` (every positional argument except a trailing Cyrillic one, which must equal the page title), `{{mn-noun|<script>}}`, `{{mn-verb|..}}`, `{{mn-adj|..}}`, `{{mn-adv|..}}`, `{{mn-propn|..}}`, `{{mn-head|<pos>|<script>}}`, plus the named `mn=`/`cn=` arguments of `mn-variant`. Script strings inside `{{inh}}`, `{{cog}}`, `{{m}}`, `{{der}}`, `{{desc}}`, `{{l}}` etc. are etymology or cross-reference data and were never used.

| source templates (all agreeing on one string) | kept rows |
|---|---|
| mn-noun | 887 |
| mn-variant | 686 |
| mn-verb | 247 |
| mn-adj | 103 |
| mn-head | 41 |
| mn-adv | 20 |
| mn-adj+mn-noun | 17 |
| mn-noun+mn-variant | 14 |
| mn-adj+mn-head+mn-noun | 3 |
| mn-adj+mn-head | 2 |
| mn-head+mn-noun | 2 |
| mn-noun+mn-verb | 2 |
| mn-adj+mn-adv+mn-noun | 2 |
| mn-adv+mn-head | 1 |
| mn-adv+mn-noun | 1 |
| mn-adj+mn-variant | 1 |

`mn-variant`, `mn-noun`, `mn-verb`, `mn-adj`, `mn-head` and `mn-adv` are the only templates whose arguments were found to carry the script form of the Cyrillic headword; the source Lua modules (`Module:mn-variant`, `Module:mn-headword`) were read to confirm the argument meaning (headword templates: the other-script spelling; `mn-variant`: Mongolian script, with `mn=`/`cn=` for Mongolia/China splits).

## 10 random kept entries (seed 7)

| cyrillic | traditional | pos | revision | flag | source |
|---|---|---|---|---|---|
| зардал | ᠵᠠᠷᠤᠳᠠᠯ (`ᠵᠠᠷᠤᠳᠠᠯ`) | Noun | 48109286 | no | mn-noun |
| үйлдэржилт | ᠦᠢᠯᠡᠳᠪᠦᠷᠢᠵᠢᠯᠲᠠ (`ᠦᠢᠯᠡᠳᠪᠦᠷᠢᠵᠢᠯᠲᠠ`) | Noun | 79393447 | no | mn-noun |
| бярга | ᠪᠢᠷᠭ᠎ᠠ (`ᠪᠢᠷᠭ<U+180E>ᠠ`) | Noun | 63039703 | yes | mn-noun |
| лоозон | ᠯᠣᠽᠦᠩ (`ᠯᠣᠽᠦᠩ`) | Noun | 92443874 | no | mn-noun |
| төмөр | ᠲᠡᠮᠦᠷ (`ᠲᠡᠮᠦᠷ`) | Noun | 87333305 | no | mn-variant |
| араа | ᠠᠷᠠᠭ᠎ᠠ (`ᠠᠷᠠᠭ<U+180E>ᠠ`) | Noun | 92444828 | yes | mn-noun |
| баг | ᠪᠠᠭ (`ᠪᠠᠭ`) | Noun | 95427100 | no | mn-noun |
| хөөрөг | ᠭᠦᠭᠦᠷᠭᠡ (`ᠭᠦᠭᠦᠷᠭᠡ`) | Noun | 92444692 | no | mn-noun |
| сайд | ᠰᠠᠢᠳ (`ᠰᠠᠢᠳ`) | Noun | 92946869 | no | mn-variant |
| баримал | ᠪᠠᠷᠢᠮᠠᠯ (`ᠪᠠᠷᠢᠮᠠᠯ`) | Noun | 90899950 | no | mn-noun |

## Independent check through the rendered HTML (action=parse)

15 of 15 randomly chosen kept entries (seed 20261107) had their script string present in the rendered `Mongolian` section of the page (section located by the `h2#Mongolian` heading up to the next language heading; substring match on the unescaped HTML and on the tag-stripped text).

| cyrillic | agrees | occurrences in rendered text |
|---|---|---|
| суулгах | yes | 1 |
| унгах | yes | 1 |
| каталог | yes | 1 |
| мянган | yes | 1 |
| хөдөлгүүр | yes | 1 |
| америк | yes | 1 |
| бялт | yes | 1 |
| хойтон | yes | 2 |
| тогтоомж | yes | 1 |
| задгай | yes | 1 |
| диваажин | yes | 1 |
| өчүүхэн | yes | 2 |
| гэнэт | yes | 1 |
| хавчих | yes | 1 |
| баатар | yes | 3 |

This confirms the extraction matches what Wiktionary renders; it is not an independent confirmation that the Wiktionary spelling itself is right.

## Patterns and caveats

1. **Volume above the aim.** 2029 entries passed every rule, about twice the 300-1000 target. Nothing was dropped at random; the file is sorted by Cyrillic, so a deterministic subset is easy to cut. The `source_templates` column (an extra last column) allows filtering by provenance.
2. **The forms are dictionary (etymological) spellings, not letter-for-letter transliterations.** Script length divided by Cyrillic length has median 1.17, range 0.60-2.67. Examples: тэнгэр -> ᠲᠩᠷᠢ, бодит -> ᠪᠣᠳᠠᠲᠤ, одоо -> ᠣᠳᠤ, лаа -> ᠯᠠ. A rule-based transliterator will legitimately differ on omitted/added vowels, so match rates against this list measure agreement with conventional spelling, not rule correctness alone.
3. **Shared script strings.** 12 script strings occur under more than one Cyrillic title (25 rows): ᠪᠤᠢ <- буй, бэ, вэ; ᠭᠠᠷᠠᠰᠢ <- гарааш, гараш; ᠭᠠᠷᠠᠵᠢ <- гараж, граж; ᠭᠦᠭᠦᠷᠭᠡ <- гүүр, хөөрөг; ᠥᠨᠦᠴᠢᠨ <- енчин, өнчин; ᠣᠳᠤ <- од, одоо; ᠰᠤᠪᠤᠷᠭ᠎ᠠ <- суварга, суврага; ᠰᠦᠮᠣ᠋ <- сумо, сүмо; ᠲᠠᠮᠤᠬᠤ <- тамах, томох; ᠬᠠᠰ <- хас, хаш; ᠬᠦᠮᠦᠨ <- хүмүүн, хүн; ᠴᠣᠬᠤ <- цох, шохой. Most are genuine homographs (од/одоо, хас/хаш, буй/бэ/вэ); `гүүр`/`хөөрөг` and `цох`/`шохой` look suspicious and deserve a manual look.
4. **Disagreeing `mn-variant` arguments.** 52 pages were rejected because the positional arguments of `mn-variant` disagree with each other. In the 3-argument form `{{mn-variant|A|B|cyrillic}}` the current module reads only argument 1 and does not document argument 2. The differences are mixed: a few are letter-choice pairs (U+183B vs U+183A, as in академи ᠠᠻᠠᠳᠧᠮᠢ / ᠠᠺᠠᠳᠧᠮᠢ; U+182C vs U+183E, as in архив) that fit Mongolia-vs-China conventions, but most are extra or missing U+202F, a different vowel, or a different word. Whether argument 2 is the China form is an inference only. All of these were rejected under the exactly-one-string rule.
5. **Whole-page rejection for partial labels.** A page such as гудамж (first sense unlabelled, three senses labelled China) is rejected as a whole.
6. **Part of speech is page-wide.** `part_of_speech` lists every POS heading in the Mongolian section (joined by `/`), not only the POS of the template that carried the script.
7. **Independence.** The Wiktionary documentation of `mn-variant` itself warns that another wiki's Mongolian-script data contains many incorrect transliterations; only the English Wiktionary was used here. I cannot rule out that some English-Wiktionary editors derived spellings mechanically; nothing in the wikitext distinguishes such entries.
8. **Revisions.** Each row carries the revision id of the wikitext that was parsed; pages can have changed since.
9. **Rate limiting.** The API answered HTTP 429 during the first title run (three in a row, so that run stopped as designed). The client was then changed to wait the server's Retry-After and to use a 1.0 s interval; the later runs completed. All 2689 pages were fetched.

