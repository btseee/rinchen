# DECISIONS.md

Each entry is tagged `[EVIDENCE: source]` (verified fact) or `[DESIGN: rationale]`
(a choice we made). Full backing detail is in
`docs/research/2026-10-28-python314-stack-and-prompt.md`.

- Pin `eole==0.6.4`, not the GitHub-tagged 0.6.5. [EVIDENCE: 0.6.4 is on PyPI;
  0.6.5 was only tagged on GitHub as of 2026-10-28]. Upgrade only after Phase 0's
  parity smoke test passes on 0.6.5.
- Build our own COMET-22 inference instead of using `unbabel-comet`. [EVIDENCE:
  unbabel-comet 2.2.7 pins `transformers<5`, incompatible with our transformers
  5.21.0 pin].
- Build our own MetricX-24 inference instead of the official `metricx` repo.
  [EVIDENCE: official code pins `transformers==4.30.2`].
- Build Korean `ko-mecab` tokenizer from source (python-mecab-ko sdist) for BLEU;
  fall back to `kiwipiepy`-based `BLEU[ko-kiwi]` labeled non-comparable if the
  build fails. [DESIGN: no cp314 wheel found for python-mecab-ko as of 2026-10-28].
- Use `position_encoding_type: SinusoidalInterleaved` in every Eole model, never
  Relative/rotary/ALiBi. [EVIDENCE: CTranslate2 issue #2134 (open as of
  2026-10-27) shows the Eole→CT2 converter mis-scales relative position keys;
  CT2 also rejects rotary for encoder-decoder].
- Do not include vLLM in the main uv workspace. [EVIDENCE: vllm==0.32.0 requires
  torch==2.13.0 and transformers<5.20.0, which conflicts with the base
  torch==2.14.1/transformers==5.21.0 pins; uv shares one lockfile across all
  dependency groups, so an in-workspace group cannot isolate this]. If a
  generator model is needed (Phase 6), use a separate non-workspace uv project
  with its own lockfile, or the llama.cpp path the research doc calls
  3.14-safe.
- Track manifests/*.parquet in git rather than ignoring it. [DESIGN: the hard
  constraints require recording resolved versions and checkpoint hashes in
  manifests/, and later phases (release.json) depend on those hashes being
  present in a fresh clone].
