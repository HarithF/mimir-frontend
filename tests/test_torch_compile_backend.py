"""End-to-end tests for the "mimir" torch.compile backend.

Each test lowers a Dynamo graph through MimIR to a shared library (requires
clang on PATH) and compares the JIT-compiled result against eager PyTorch.
"""

import shutil

import pytest
import torch

from mimir_frontend.backend import mimir_backend

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")


class LinearMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SmallConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3)
        self.fc = torch.nn.Linear(4 * 6 * 6, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


@pytest.fixture(autouse=True)
def _reset_dynamo(monkeypatch, tmp_path_factory):
    # Hermetic per-test-run JIT cache: don't read from or pollute the user cache.
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "mimir-jit-cache"))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def check_against_eager(model, *inputs):
    with torch.no_grad():
        want = model(*inputs)
        compiled = torch.compile(model, backend=mimir_backend)
        got = compiled(*inputs)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_linear_mlp_matches_eager():
    check_against_eager(LinearMLP(), torch.randn(4, 16))


def test_conv_matches_eager():
    check_against_eager(SmallConv(), torch.randn(2, 1, 8, 8))


def test_registered_by_name():
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        want = model(x)
        got = torch.compile(model, backend="mimir")(x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


class TwoOutputs(torch.nn.Module):
    """Two outputs of different shapes sharing an intermediate."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32)
        self.head_a = torch.nn.Linear(32, 8)
        self.head_b = torch.nn.Linear(32, 3)

    def forward(self, x):
        hidden = torch.relu(self.fc(x))
        return self.head_a(hidden), self.head_b(hidden)


class ThreeOutputs(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.relu(x - y), (x * y).sum(dim=1)


def test_two_outputs_match_eager():
    model = TwoOutputs()
    x = torch.randn(4, 16)
    with torch.no_grad():
        want = model(x)
        got = torch.compile(model, backend="mimir")(x)
    assert isinstance(got, tuple) and len(got) == 2
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_three_outputs_match_eager():
    model = ThreeOutputs()
    x, y = torch.randn(4, 6), torch.randn(4, 6)
    with torch.no_grad():
        want = model(x, y)
        got = torch.compile(model, backend="mimir")(x, y)
    assert isinstance(got, tuple) and len(got) == 3
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_cache_hit_reuses_compiled_so(tmp_path):
    x = torch.randn(4, 16)

    model_a = LinearMLP()
    with torch.no_grad():
        got_a = torch.compile(model_a, backend="mimir", options={"cache_dir": str(tmp_path)})(x)
        torch.testing.assert_close(got_a, model_a(x), rtol=1e-4, atol=1e-4)

    so_files = list(tmp_path.glob("*.so"))
    assert len(so_files) == 1
    first_mtime = so_files[0].stat().st_mtime_ns

    # A second model with the same architecture but different weights hits the
    # same cache entry: weights are runtime arguments, not part of the key.
    torch._dynamo.reset()
    model_b = LinearMLP()
    with torch.no_grad():
        got_b = torch.compile(model_b, backend="mimir", options={"cache_dir": str(tmp_path)})(x)
        torch.testing.assert_close(got_b, model_b(x), rtol=1e-4, atol=1e-4)

    so_files = list(tmp_path.glob("*.so"))
    assert len(so_files) == 1
    assert so_files[0].stat().st_mtime_ns == first_mtime, "cache entry was rebuilt instead of reused"
    assert not torch.allclose(got_a, got_b), "different weights must give different results"


def test_cache_can_be_disabled(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        got = torch.compile(model, backend="mimir", options={"cache": False, "cache_dir": str(tmp_path)})(x)
        torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-4)
    assert not list(tmp_path.glob("*.so"))


def test_profile_summary_prints_report(tmp_path, capsys):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(
            model, backend="mimir", options={"profile": "summary", "cache_dir": str(tmp_path)}
        )
        got = compiled(x)
        torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-4)
    assert "Phase profile (flat):" in capsys.readouterr().err


def test_profile_trace_writes_json(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(
            model,
            backend="mimir",
            options={"profile": "trace", "debug_dir": str(tmp_path), "cache_dir": str(tmp_path)},
        )
        compiled(x)
    traces = list(tmp_path.glob("*_profile.json"))
    assert traces and "traceEvents" in traces[0].read_text()


def test_debug_dir_dumps_artifacts(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(model, backend="mimir", options={"debug_dir": str(tmp_path)})
        got = compiled(x)
        want = model(x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    for suffix in ("_pre.mim", "_post.mim", ".ll", ".so"):
        matches = list(tmp_path.glob(f"mimir_graph_*{suffix}"))
        assert matches, f"expected a mimir_graph_*{suffix} artifact in {tmp_path}"


def _emitted_fp_ops(tmp_path):
    """The distinct `fadd`/`fmul`/... forms in the .ll dumped under `tmp_path`."""
    import re

    ll = next(iter(tmp_path.glob("mimir_graph_*.ll"))).read_text()
    return set(re.findall(r"^\s*%\S+ = (f(?:add|sub|mul|div)[a-z ]*) float", ll, re.M))


def _compile_with(model, x, tmp_path, **options):
    with torch.no_grad():
        compiled = torch.compile(
            model,
            backend="mimir",
            options={"debug_dir": str(tmp_path), "cache_dir": str(tmp_path), **options},
        )
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)


def test_fast_math_is_on_by_default(tmp_path):
    _compile_with(LinearMLP(), torch.randn(4, 16), tmp_path)
    assert _emitted_fp_ops(tmp_path) == {"fadd fast", "fmul fast"}


def test_fast_math_can_be_disabled(tmp_path):
    _compile_with(LinearMLP(), torch.randn(4, 16), tmp_path, fast_math=False)
    assert _emitted_fp_ops(tmp_path) == {"fadd", "fmul"}


def test_fast_math_accepts_individual_flags(tmp_path):
    _compile_with(LinearMLP(), torch.randn(4, 16), tmp_path, fast_math="contract|reassoc")
    assert _emitted_fp_ops(tmp_path) == {"fadd contract reassoc", "fmul contract reassoc"}


def test_fast_math_follows_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMIR_FAST_MATH", "nnan|contract")
    _compile_with(LinearMLP(), torch.randn(4, 16), tmp_path)
    assert _emitted_fp_ops(tmp_path) == {"fadd nnan contract", "fmul nnan contract"}


def test_fast_math_rejects_unknown_flag():
    # Dynamo re-raises a backend failure as BackendCompilerFailed, keeping the message.
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed, match="unknown fast-math flag"):
        with torch.no_grad():
            torch.compile(LinearMLP(), backend="mimir", options={"fast_math": "bogus"})(
                torch.randn(4, 16)
            )


def test_fast_math_separates_cache_entries(tmp_path):
    """The same graph at two modes must not share one cached library."""
    model, x = LinearMLP(), torch.randn(4, 16)
    with torch.no_grad():
        for fast_math in (True, False):
            torch._dynamo.reset()
            compiled = torch.compile(
                model, backend="mimir", options={"cache_dir": str(tmp_path), "fast_math": fast_math}
            )
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)

    assert len({p.name for p in tmp_path.glob("*.so")}) == 2


class Bmm(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


class Matmul(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


class GroupedConv(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 12, 3, padding=1, groups=groups)

    def forward(self, x):
        return self.conv(x)


def test_bmm_matches_eager():
    check_against_eager(Bmm(), torch.randn(3, 4, 5), torch.randn(3, 5, 2))


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        ((4, 8), (8, 3)),        # 2-D x 2-D
        ((2, 4, 8), (2, 8, 3)),  # batched, matching prefix
        ((2, 4, 8), (8, 3)),     # batched x plain matrix
        ((2, 3, 4, 8), (8, 5)),  # two batch dims
        ((1, 4, 8), (2, 8, 3)),  # broadcast the batch prefix
        ((8,), (8, 3)),          # 1-D lhs is promoted then dropped
        ((4, 8), (8,)),          # 1-D rhs is promoted then dropped
    ],
)
def test_matmul_matches_eager(lhs_shape, rhs_shape):
    check_against_eager(Matmul(), torch.randn(*lhs_shape), torch.randn(*rhs_shape))


@pytest.mark.parametrize("groups", [2, 3, 6])
def test_grouped_conv_matches_eager(groups):
    """`groups == 6` is the depthwise case, where every channel is its own group."""
    check_against_eager(GroupedConv(groups), torch.randn(2, 6, 8, 8))


class BatchNormNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class BatchNormNoAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4, affine=False)

    def forward(self, x):
        return self.bn(x)


class BatchedLinear(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.fc = torch.nn.Linear(8, 5, bias=bias)

    def forward(self, x):
        return self.fc(x)


def test_batch_norm_matches_eager():
    model = BatchNormNet().eval()
    # Give the running statistics something other than their 0/1 initialization.
    with torch.no_grad():
        model.bn.running_mean.normal_()
        model.bn.running_var.uniform_(0.5, 2.0)
    check_against_eager(model, torch.randn(2, 3, 8, 8))


def test_batch_norm_without_affine_matches_eager():
    model = BatchNormNoAffine().eval()
    with torch.no_grad():
        model.bn.running_mean.normal_()
        model.bn.running_var.uniform_(0.5, 2.0)
    check_against_eager(model, torch.randn(2, 4, 5, 5))


@pytest.mark.parametrize("shape", [(4, 8), (2, 4, 8), (2, 3, 4, 8)])
@pytest.mark.parametrize("bias", [True, False])
def test_batched_linear_matches_eager(shape, bias):
    check_against_eager(BatchedLinear(bias).eval(), torch.randn(*shape))
