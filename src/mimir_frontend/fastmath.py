"""Floating-point relaxation modes for the `%math` axms the frontend emits.

Every `%math` operation takes a `Nat` mode just in front of its actual argument
(see `MimIR/src/mim/plug/math/math.mim`). It drives two things:

* MimIR's own `normalize_arith`/`normalize_extrema`, which only apply algebraic
  identities (`0 + b -> b`, `1 * b -> b`, ...) for `Mode.fast`, and
* the `ll` backend, which turns the bits into LLVM fast-math flags, so
  `fadd float` becomes `fadd fast float` (or `fadd contract reassoc float`).

"""

from __future__ import annotations

from enum import IntFlag
import os
import re
from collections.abc import Iterable

__all__ = ["Mode", "DEFAULT_MODE", "default_mode", "parse_mode", "mode_name"]


class Mode(IntFlag):
    """Allowed FP optimizations, mirroring `mim::plug::math::Mode`.

    Keep the bit values in sync with `MimIR/include/mim/plug/math/math.h`.
    """

    none = 0
    nnan = 1 << 0  # assume no NaN operands/results
    ninf = 1 << 1  # assume no +/-Inf operands/results
    nsz = 1 << 2  # the sign of a zero is insignificant
    arcp = 1 << 3  # allow x / y -> x * (1 / y)
    contract = 1 << 4  # allow fusing a mul+add into an FMA
    afn = 1 << 5  # allow approximate sin/log/sqrt/... implementations
    reassoc = 1 << 6  # allow reassociating FP expressions

    finite = nnan | ninf
    unsafe = nsz | arcp | reassoc
    fast = nnan | ninf | nsz | arcp | contract | afn | reassoc


#: Mode used when neither the caller nor `MIMIR_FAST_MATH` says otherwise.
DEFAULT_MODE = Mode.fast

_ENV_VAR = "MIMIR_FAST_MATH"

_TRUE = {"1", "on", "true", "yes", "y", "fast", "all"}
_FALSE = {"0", "off", "false", "no", "n", "none", "strict", "ieee"}

_SEPARATORS = re.compile(r"[|,+\s]+")


def default_mode() -> Mode:
    """The mode to use when a caller passes no explicit setting.

    Reads `MIMIR_FAST_MATH` on every call so it can be flipped per process run
    (`MIMIR_FAST_MATH=off uv run ...`) without touching any call site.
    """
    env = os.environ.get(_ENV_VAR)
    if env is None or not env.strip():
        return DEFAULT_MODE
    return _parse(env, source=f"{_ENV_VAR}=")


def parse_mode(value: object) -> Mode:
    """Coerce a user-supplied fast-math setting into a `Mode`.

    Accepts `None` (-> `default_mode()`), a `bool` (`Mode.fast`/`Mode.none`), a
    raw bit set, a `Mode`, a flag name or a `"contract|reassoc"`-style list of
    them, or any iterable of those.
    """
    if value is None:
        return default_mode()
    return _parse(value, source="fast_math=")


def mode_name(mode: Mode) -> str:
    """Readable rendering of `mode`, e.g. `"fast"`, `"none"`, `"contract|nnan"`."""
    mode = Mode(mode)
    if mode == Mode.none:
        return "none"
    if mode == Mode.fast:
        return "fast"
    # Iterating a flag instance yields exactly its set canonical (single-bit) flags.
    return "|".join(flag.name for flag in mode)


def _parse(value: object, *, source: str) -> Mode:
    if isinstance(value, Mode):
        return value
    if isinstance(value, bool):
        return Mode.fast if value else Mode.none
    if isinstance(value, int):
        return _from_int(value, source=source)
    if isinstance(value, str):
        return _from_str(value, source=source)
    if isinstance(value, Iterable):
        mode = Mode.none
        for item in value:
            mode |= _parse(item, source=source)
        return mode
    raise TypeError(
        f"{source}{value!r}: expected None, bool, int, str, Mode, or an iterable of those"
    )


def _from_int(value: int, *, source: str) -> Mode:
    if not 0 <= value <= int(Mode.fast):
        raise ValueError(f"{source}{value}: mode bits must be in [0, {int(Mode.fast)}]")
    return Mode(value)


def _from_str(value: str, *, source: str) -> Mode:
    text = value.strip()
    if not text:
        return DEFAULT_MODE

    lowered = text.lower()
    if lowered in _TRUE:
        return Mode.fast
    if lowered in _FALSE:
        return Mode.none
    try:
        return _from_int(int(lowered, 0), source=source)
    except ValueError as exc:
        if "mode bits must be" in str(exc):
            raise

    mode = Mode.none
    for part in _SEPARATORS.split(lowered):
        if not part:
            continue
        flag = Mode.__members__.get(part)
        if flag is None:
            known = ", ".join(sorted(Mode.__members__))
            raise ValueError(
                f"{source}{value!r}: unknown fast-math flag {part!r} (known: {known})"
            )
        mode |= flag
    return mode
