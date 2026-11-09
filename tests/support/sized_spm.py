"""A SentencePiece model of an exact number of pieces, written without training.

Training a model of 32,768 pieces or more takes a corpus of that many distinct pieces and
minutes; the tests of the vocabulary size only need `GetPieceSize()` to say a number, so
the model is written as a protocol buffer: the three special pieces of every model and
as many ordinary pieces as the size needs, with no normaliser rules. It loads in
`sentencepiece.SentencePieceProcessor` and is not meant to encode text.
"""

from __future__ import annotations

from pathlib import Path

from sentencepiece import sentencepiece_model_pb2

# `ModelProto.SentencePiece.Type`
_UNKNOWN = 2
_CONTROL = 3
_SPECIAL_PIECES = (("<unk>", _UNKNOWN), ("<s>", _CONTROL), ("</s>", _CONTROL))


def sized_spm_bytes(pieces: int) -> bytes:
    """The bytes of a model that holds exactly `pieces` pieces (at least 3)."""
    assert pieces >= len(_SPECIAL_PIECES)
    model = sentencepiece_model_pb2.ModelProto()
    model.trainer_spec.model_type = sentencepiece_model_pb2.TrainerSpec.UNIGRAM
    model.trainer_spec.vocab_size = pieces
    for text, kind in _SPECIAL_PIECES:
        entry = model.pieces.add()
        entry.piece = text
        entry.type = kind
    number = 0
    while len(model.pieces) < pieces:
        entry = model.pieces.add()
        entry.piece = f"p{number}"
        entry.score = -float(number)
        number += 1
    return model.SerializeToString()


def write_sized_spm(path: Path, pieces: int) -> Path:
    """Write to `path` a model that holds exactly `pieces` pieces (at least 3), and
    return `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sized_spm_bytes(pieces))
    return path
