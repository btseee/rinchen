"""The request, the response and the errors of the service (D11, T15).

A request is strict (a quoted number or a boolean in the wrong place is refused, not coerced)
and refuses a field it does not know, so a client that sends `terminology` before the service
reads it learns so. Its limits are the defaults of the config; `bounded_request_model` narrows
them to the configured ones and keeps every other rule.

Every error answers `{"error": {"code": str, "message": str}}`, the 422 of a refused body
included. A message names a field and what is wrong with it and never holds what a client sent:
the text of a request is not logged, echoed or put in an exception. The one thing of the client's
a message does name is an unknown key, up to 40 characters and then cut with an ellipsis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, Literal

from fastapi.responses import JSONResponse
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StringConstraints,
    ValidationError,
    create_model,
    model_serializer,
    model_validator,
)
from pydantic_core import InitErrorDetails

from rinchen.config.serve import (
    DEFAULT_MAX_TEXT_CHARS,
    MAX_BEAM_SIZE,
    MAX_TEXT_CHARS_LIMIT,
    MAX_TEXTS_LIMIT,
)
from rinchen.errors import ConfigError
from rinchen.utils.integers import is_strict_int

# A registry code is eight to ten characters; the room is for a typo to be echoed in the
# engine's refusal, not for a text.
MAX_LANGUAGE_CODE_CHARS: Final = 32
# How much of an unknown key of a body a refusal names (no field of ours is longer).
MAX_KEY_CHARS: Final = 40


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


_LanguageField = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_LANGUAGE_CODE_CHARS),
    AfterValidator(_reject_blank),
]


_Texts = Annotated[
    list[Annotated[str, StringConstraints(max_length=DEFAULT_MAX_TEXT_CHARS)]],
    Field(min_length=1, max_length=MAX_TEXTS_LIMIT),
]


def _texts_type(max_texts: int, max_text_chars: int) -> Any:
    text = Annotated[str, StringConstraints(max_length=max_text_chars)]
    return Annotated[list[text], Field(min_length=1, max_length=max_texts)]


class TranslateRequest(BaseModel):
    """`POST /v1/translate`: the texts, the languages and how to decode them.

    `src` is a registry code or `auto` (the script hint: `mn_Mong` or `und`); `script`
    asks for a target script on a Mongolian target; `doc_mode` translates the texts as
    documents; `beam_size` is the beam of this request (4 is what the schema says when none is
    sent: the service decodes with the beam of its own config then); `n_best` is how many
    hypotheses to return, at most the beam's size. An empty text is allowed: it is answered
    with the status `empty`."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    texts: _Texts
    src: _LanguageField = "auto"
    tgt: _LanguageField
    script: Literal["Cyrl", "Mong"] | None = None
    beam_size: Annotated[int, Field(ge=1, le=MAX_BEAM_SIZE)] = 4
    n_best: Annotated[int, Field(ge=1)] = 1
    doc_mode: bool = False

    @model_validator(mode="after")
    def _within_the_beam_that_was_sent(self) -> TranslateRequest:
        # Without a `beam_size` the service decodes with the beam of its config, which a schema
        # cannot know: `serve._translate.decode_options` judges `n_best` against that one. A
        # beam that failed its own check never gets here and is reported once, under its name.
        if "beam_size" in self.model_fields_set and self.n_best > self.beam_size:
            message = (
                f"n_best ({self.n_best}) must not exceed beam_size ({self.beam_size}): "
                "the beam holds no more hypotheses"
            )
            # Raised as an error of the field, so that the refusal names `n_best`.
            raise ValidationError.from_exception_data(
                type(self).__name__,
                [
                    InitErrorDetails(
                        type="value_error",
                        loc=("n_best",),
                        input=self.n_best,
                        ctx={"error": ValueError(message)},
                    )
                ],
            )
        return self


def bounded_request_model(
    *, max_texts: int, max_text_chars: int
) -> type[TranslateRequest]:
    """`TranslateRequest` with the limits of a service: at most `max_texts` texts of at most
    `max_text_chars` characters each. It is a subclass of the same name (so the interface
    description keeps one), and a refusal names `texts` or `texts.<index>`.

    Raises:
        ConfigError: a limit is not an integer in the range the config allows.
    """
    for name, value, highest in (
        ("max_texts", max_texts, MAX_TEXTS_LIMIT),
        ("max_text_chars", max_text_chars, MAX_TEXT_CHARS_LIMIT),
    ):
        if not is_strict_int(value) or not 1 <= value <= highest:
            raise ConfigError(f"{name} must be between 1 and {highest}, got {value!r}")
    return create_model(
        "TranslateRequest",
        __base__=TranslateRequest,
        __module__=__name__,
        texts=(_texts_type(max_texts, max_text_chars), ...),
    )


class TranslateResponse(BaseModel):
    """What a translate request answers, one entry per text in order: `translations`,
    `detected_src` (the source language, or what `auto` found), the engine's `statuses`
    (`ok`, `empty`, `too_long`, `truncated`, `no_output`: a status other than `ok` is
    still a 200) and `script_purity` (`null` where there is no judgement). `alternatives`
    appears only for a request that asked for more than one hypothesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    translations: list[str]
    detected_src: list[str]
    model: str
    latency_ms: Annotated[float, Field(ge=0)]
    statuses: list[str]
    script_purity: list[float | None]
    alternatives: list[list[str]] | None = None

    @model_serializer(mode="wrap")
    def _leave_out_absent_alternatives(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        body: dict[str, Any] = handler(self)
        if self.alternatives is None:
            body.pop("alternatives", None)
        return body


class ErrorBody(BaseModel):
    """The `error` member: a short machine code and a message for a person."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str


class ErrorResponse(BaseModel):
    """The one shape of every error the service answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorBody


def error_response(status: int, code: str, message: str) -> JSONResponse:
    """A response of `status` whose body is `{"error": {"code": code, "message": message}}`."""
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump())


def validation_message(errors: Sequence[Mapping[str, Any]]) -> str:
    """The complaints of a refused body, each as `<field>: <what is wrong>` and joined by
    `; `: from `loc` and `msg` only, never from the `input` the errors also carry."""
    return "; ".join(f"{_field_of(error)}: {error['msg']}" for error in errors)


def _field_of(error: Mapping[str, Any]) -> str:
    location = tuple(error.get("loc") or ())
    # The body is the only input, so its name says nothing before a field's.
    if len(location) > 1 and location[0] == "body" and isinstance(location[1], str):
        location = location[1:]
    return ".".join(_shown(part) for part in location) or "body"


def _shown(part: str | int) -> str:
    """A part of a location as the message says it. The name of an unknown key is the client's
    own, so it is cut after `MAX_KEY_CHARS` characters: the client still learns which field is
    unknown, and no key is echoed whole whatever its length."""
    name = str(part)
    if len(name) <= MAX_KEY_CHARS:
        return name
    return name[:MAX_KEY_CHARS] + "..."


def validation_error_response(errors: Sequence[Mapping[str, Any]]) -> JSONResponse:
    """The 422 of a body that failed validation, in the error shape: `errors` is what
    pydantic's `ValidationError.errors()` or FastAPI's `RequestValidationError.errors()`
    returns."""
    return error_response(422, "validation_error", validation_message(errors))
