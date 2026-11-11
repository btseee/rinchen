"""What the text files of a release must not hold (D13, spec T18): a token and a local path.

A release is uploaded whole and is read by anyone who may read the repository, so `.json` and `.md`
files are scanned before the upload is planned. `kinds_in` returns the kinds of problem a text holds
and never the text that matched: a message about a file names the file and the kind.

- A token: `hf_` followed by 20 or more letters or digits, `Bearer` and 16 or more token characters,
  or the assignment of `HF_TOKEN`: the name, an optional quote (a backslash before it allowed, as in a
  JSON string inside a JSON string), optional blanks and `=` or `:` (`HF_TOKEN=x`, `"HF_TOKEN": "x"`).
- An absolute local path: a POSIX path of at least two components whose first slash starts a word
  (after a blank, a quote of any kind, a bracket, `=`, `,`, `;`, `:` or `|`, or straight after the
  `://` of an address: `file:///home/x/y`), under any root (`/home`, `/tmp`, `/var`, `/root`,
  `/Users`, `/mnt`, `/opt`, `/usr`, `/srv`, ...), or a drive path (`X:` and a backslash). The export's
  own `manifest.json` holds such paths, which is why it is never uploaded. A slash inside a word
  (`and/or`, `org/name`, `path/to/x`) or in the path of an address with a host (`https://host/x`) is
  not a local path.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["KIND_PATH", "KIND_TOKEN", "kinds_in"]

KIND_TOKEN: Final = "looks like a token"
KIND_PATH: Final = "holds an absolute local path"

_TOKEN_PATTERNS: Final = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i:bearer)\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"HF_TOKEN\\?[\"']?\s*[:=]"),
)
# The characters after which a slash starts a path: blanks, quotes (straight and typographic),
# brackets and separators; `://` (an address with no host) is the other way in.
_PATH_PATTERNS: Final = (
    re.compile(
        r"(?:^|(?<=://)|[\s\"'`\u2018\u2019\u201c\u201d\u00ab\u00bb(\[{<=,;:|])"
        r"/[A-Za-z0-9._~-]+/[A-Za-z0-9._~-]"
    ),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\"),
)


def kinds_in(data: bytes) -> list[str]:
    """The kinds of problem the text `data` (UTF-8; a byte that is not is read as a replacement
    character) holds, each once, a token first."""
    text = data.decode("utf-8", errors="replace")
    kinds: list[str] = []
    if any(pattern.search(text) for pattern in _TOKEN_PATTERNS):
        kinds.append(KIND_TOKEN)
    if any(pattern.search(text) for pattern in _PATH_PATTERNS):
        kinds.append(KIND_PATH)
    return kinds
