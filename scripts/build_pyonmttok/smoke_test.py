#!/usr/bin/env python
"""Round-trip smoke test for a built pyonmttok wheel (from its own README)."""
import sys


def main() -> int:
    assert sys.version_info[:2] == (3, 14), f"expected cp314, got {sys.version}"

    import pyonmttok

    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
    tokens, _ = tokenizer.tokenize("Hello World!")
    detokenized = tokenizer.detokenize(tokens)
    assert detokenized == "Hello World!", f"round-trip mismatch: {detokenized!r}"

    print("OK: pyonmttok import and tokenize/detokenize round-trip succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
