"""Fast-math modes: parsing, propagation into %math axms, and LLVM flag emission."""

import pytest
import torch
from torch import fx

import mim

from mimir_frontend.fastmath import (
    DEFAULT_MODE,
    Mode,
    default_mode,
    mode_name,
    parse_mode,
)
from mimir_frontend.operators import OperatorLibrary
from mimir_frontend.utils import model_to_mimir


def make_world() -> mim.World:
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "affine"])
    return driver.world()


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.fc(x))


# --- Mode bits stay in sync with mim::plug::math::Mode -----------------------


def test_mode_bits_match_mimir():
    assert (int(Mode.nnan), int(Mode.ninf), int(Mode.nsz), int(Mode.arcp)) == (
        1,
        2,
        4,
        8,
    )
    assert (int(Mode.contract), int(Mode.afn), int(Mode.reassoc)) == (16, 32, 64)
    assert int(Mode.none) == 0
    assert int(Mode.finite) == int(Mode.nnan | Mode.ninf)
    assert int(Mode.unsafe) == int(Mode.nsz | Mode.arcp | Mode.reassoc)
    assert int(Mode.fast) == 127


# --- parse_mode --------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, DEFAULT_MODE),
        (True, Mode.fast),
        (False, Mode.none),
        ("fast", Mode.fast),
        ("FAST", Mode.fast),
        ("on", Mode.fast),
        ("off", Mode.none),
        ("none", Mode.none),
        ("strict", Mode.none),
        ("", DEFAULT_MODE),
        ("contract", Mode.contract),
        ("contract|reassoc", Mode.contract | Mode.reassoc),
        ("contract, nnan", Mode.contract | Mode.nnan),
        ("contract reassoc", Mode.contract | Mode.reassoc),
        ("finite", Mode.finite),
        (16, Mode.contract),
        ("16", Mode.contract),
        ("0x10", Mode.contract),
        (Mode.afn, Mode.afn),
        (["contract", "afn"], Mode.contract | Mode.afn),
    ],
)
def test_parse_mode(value, expected):
    assert parse_mode(value) == expected


@pytest.mark.parametrize("value", ["bogus", "contract|bogus"])
def test_parse_mode_rejects_unknown_flag(value):
    with pytest.raises(ValueError, match="unknown fast-math flag"):
        parse_mode(value)


@pytest.mark.parametrize("value", [-1, 128, 999])
def test_parse_mode_rejects_out_of_range_bits(value):
    with pytest.raises(ValueError, match="mode bits must be"):
        parse_mode(value)


def test_parse_mode_rejects_wrong_type():
    with pytest.raises(TypeError):
        parse_mode(1.5)


def test_mode_name_roundtrips():
    for mode in [Mode.none, Mode.fast, Mode.finite, Mode.contract | Mode.reassoc]:
        assert parse_mode(mode_name(mode)) == mode


def test_default_mode_follows_env(monkeypatch):
    monkeypatch.delenv("MIMIR_FAST_MATH", raising=False)
    assert default_mode() == DEFAULT_MODE
    monkeypatch.setenv("MIMIR_FAST_MATH", "contract|reassoc")
    assert default_mode() == Mode.contract | Mode.reassoc
    assert parse_mode(None) == Mode.contract | Mode.reassoc
    # An explicit setting always wins over the environment.
    assert parse_mode(False) == Mode.none
    monkeypatch.setenv("MIMIR_FAST_MATH", "   ")
    assert default_mode() == DEFAULT_MODE


# --- the mode reaches the %math axms ----------------------------------------


@pytest.mark.parametrize(
    "math_mode,expected",
    [
        (None, DEFAULT_MODE),
        (False, Mode.none),
        ("contract|reassoc", Mode.contract | Mode.reassoc),
    ],
)
def test_operator_library_binds_requested_mode(math_mode, expected):
    ops = OperatorLibrary(make_world(), math_mode=math_mode)
    assert ops.math_mode == expected
    # Every %math axm is curried with this Nat, so checking the literal that all
    # of them were bound with covers add/mul/cmp/exp/... in one go.
    assert ops.math_mode_lit == ops.world.lit_nat(int(expected))


def test_operator_library_defaults_to_fast():
    assert OperatorLibrary(make_world()).math_mode == Mode.fast


@pytest.mark.parametrize(
    "math_mode,expected_bits",
    [(None, 127), (True, 127), (False, 0), ("contract|reassoc", 80), (Mode.finite, 3)],
)
def test_model_to_mimir_emits_mode(math_mode, expected_bits):
    ir = model_to_mimir(TinyMLP(), [(2, 4)], name="tiny", math_mode=math_mode)
    assert f"%math.arith.add (23, 8) {expected_bits}" in ir
    if expected_bits != 0:
        assert "%math.arith.add (23, 8) 0" not in ir


def test_model_to_mimir_mode_follows_env(monkeypatch):
    monkeypatch.setenv("MIMIR_FAST_MATH", "none")
    assert "%math.arith.add (23, 8) 0" in model_to_mimir(
        TinyMLP(), [(2, 4)], name="tiny"
    )
