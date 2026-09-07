#!/usr/bin/env python3
"""
Sweep a directory of model files (each defining Model / get_inputs / get_init_inputs)
through torch.compile(backend="mimir") and report pass/fail,
without modifying the model files themselves.

Each model is run in its OWN subprocess so that a crash / OOM-kill in one model can't take down the whole sweep


Usage:
    uv run --no-sync python compile_all_kernel.py level3
    uv run --no-sync python compile_all_kernel.py level3 --pattern "VGG"
    uv run --no-sync python compile_all_kernel.py level3 --timeout 300 --max-memory-gb 16
    uv run --no-sync python compile_all_kernel.py level3 --emit-mlir
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

# Substrings that indicate mimir hit an internal problem WITHOUT raising a
# Python exception (e.g. printed "unhandled App axiom ..." and limped on).
# If any of these show up in captured stdout/stderr during compile+run,
# we treat the model as a FAIL even if the numeric output happened to match
# (which can happen via dynamo's silent eager-fallback path).
SUSPICIOUS_MARKERS = [
    "unhandled",
    "unimplemented",
    "not supported",
    "unsupported",
]


class _Tee:
    """Write to multiple streams at once (so we can capture output for
    scanning while still printing it live to the real terminal)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def load_module(path: Path):
    """Import a model file (e.g. '11_VGG16.py') as a standalone module."""
    mod_name = f"_level3_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def outputs_close(a, b, atol, rtol):
    """Recursively compare (possibly nested) tensor / tuple / list outputs."""
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a, b, atol=atol, rtol=rtol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(outputs_close(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(outputs_close(a[k], b[k], atol, rtol) for k in a)
    return a == b


def run_one(
    path: Path,
    debug_root: Path,
    atol: float,
    rtol: float,
    fullgraph: bool,
    strict: bool,
    emit_mlir: bool = False,
):
    """Runs a single model file in THIS process. Returns a result dict.
    Only ever called inside a child subprocess (see --single below), so if
    it OOMs or hangs, it takes down only that subprocess.
    """
    import torch
    import mimir_frontend.backend  # noqa: F401  (registers the "mimir" backend)

    result = {"file": path.name, "status": None, "error_stage": None, "detail": ""}

    try:
        module = load_module(path)
    except Exception:
        result.update(
            status="fail", error_stage="import", detail=traceback.format_exc()
        )
        return result

    try:
        init_inputs = (
            module.get_init_inputs() if hasattr(module, "get_init_inputs") else []
        )
        model = module.Model(*init_inputs)
        model.eval()
        inputs = module.get_inputs()
    except Exception:
        result.update(status="fail", error_stage="setup", detail=traceback.format_exc())
        return result

    try:
        with torch.no_grad():
            want = model(*inputs)
    except Exception:
        result.update(
            status="fail", error_stage="eager_forward", detail=traceback.format_exc()
        )
        return result

    debug_dir = debug_root / path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, buf_out)),
            contextlib.redirect_stderr(_Tee(sys.stderr, buf_err)),
        ):
            compiled_model = torch.compile(
                model,
                backend="mimir",
                fullgraph=fullgraph,
                options={"debug_dir": str(debug_dir), "emit_mlir": emit_mlir},
            )
            with torch.no_grad():
                got = compiled_model(*inputs)
    except Exception:
        result.update(
            status="fail", error_stage="compile_or_run", detail=traceback.format_exc()
        )
        return result

    captured = (buf_out.getvalue() + "\n" + buf_err.getvalue()).lower()
    hit_markers = [m for m in SUSPICIOUS_MARKERS if m in captured]

    try:
        numerically_ok = outputs_close(want, got, atol, rtol)
    except Exception:
        result.update(
            status="fail", error_stage="compare", detail=traceback.format_exc()
        )
        return result

    if not numerically_ok:
        result.update(
            status="fail",
            error_stage="mismatch",
            detail="Outputs differ beyond tolerance.",
        )
    elif strict and hit_markers:
        result.update(
            status="fail",
            error_stage="backend_warning",
            detail=f"Output matched but backend logged: {hit_markers}. "
            f"Likely silent fallback rather than a real mimir compile.",
        )
    else:
        result.update(status="pass")

    return result


def _limit_memory(max_bytes):
    """preexec_fn for subprocess.Popen: cap the child's virtual address
    space so it hits a clean MemoryError instead of getting OOM-killed by
    the kernel. POSIX only; silently skipped elsewhere."""

    def _setter():
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

    return _setter


def run_child(args):
    """Entry point used when this script is invoked as a subprocess worker
    for a single model (`--single <path>`). Writes its result as JSON to
    --result-file and exits 0 (pass) or 1 (fail)."""
    path = args.single
    debug_root = args.output / "mlir"
    result = run_one(
        path,
        debug_root,
        args.atol,
        args.rtol,
        fullgraph=not args.allow_graph_breaks,
        strict=not args.no_strict,
        emit_mlir=args.emit_mlir,
    )
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_file, "w") as f:
        json.dump(result, f)
    sys.exit(0 if result["status"] == "pass" else 1)


def run_parent(args):
    """Discovers model files and runs each one in an isolated subprocess."""
    files = sorted(
        (p for p in args.dir.glob("*.py") if re.match(r"^\d+_.*\.py$", p.name)),
        key=lambda p: int(p.name.split("_", 1)[0]),
    )
    if args.pattern:
        rx = re.compile(args.pattern)
        files = [p for p in files if rx.search(p.name)]

    if not files:
        print(f"No matching model files found in {args.dir}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_memory_gb * (1024**3)) if args.max_memory_gb else None
    preexec_fn = (
        _limit_memory(max_bytes) if (max_bytes and os.name == "posix") else None
    )

    results = []
    for path in files:
        print(
            f"[{path.name}] running (subprocess, timeout={args.timeout}s)...",
            flush=True,
        )
        result_file = tmp_dir / f"{path.stem}.json"
        if result_file.exists():
            result_file.unlink()

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            str(path),
            "--result-file",
            str(result_file),
            "--output",
            str(args.output),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
        ]
        if args.allow_graph_breaks:
            cmd.append("--allow-graph-breaks")
        if args.no_strict:
            cmd.append("--no-strict")
        if args.emit_mlir:
            cmd.append("--emit-mlir")

        try:
            proc = subprocess.run(cmd, timeout=args.timeout, preexec_fn=preexec_fn)
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True

        if timed_out:
            res = {
                "file": path.name,
                "status": "fail",
                "error_stage": "timeout",
                "detail": f"Exceeded {args.timeout}s timeout; process was killed.",
            }
        elif result_file.exists():
            with open(result_file) as f:
                res = json.load(f)
        elif returncode is not None and returncode < 0:
            # Negative returncode on POSIX = killed by signal (e.g. -9 = SIGKILL,
            # almost always the OOM killer for a process like this).
            sig = -returncode
            res = {
                "file": path.name,
                "status": "fail",
                "error_stage": f"killed_signal_{sig}",
                "detail": f"Subprocess was killed by signal {sig} "
                f"(signal 9 = SIGKILL, most likely OOM). No result file written.",
            }
        else:
            res = {
                "file": path.name,
                "status": "fail",
                "error_stage": "crashed",
                "detail": f"Subprocess exited with code {returncode} and no result file.",
            }

        results.append(res)
        if res["status"] == "pass":
            print(f"[{path.name}] PASS")
        else:
            print(f"[{path.name}] FAIL ({res['error_stage']})")
            if args.verbose and res.get("detail"):
                print(res["detail"])

    passed = [r for r in results if r["status"] == "pass"]
    failed = [r for r in results if r["status"] == "fail"]

    print("\n" + "=" * 60)
    print(f"Summary: {len(passed)}/{len(results)} passed")
    if failed:
        print("\nFailures by stage:")
        by_stage = {}
        for r in failed:
            by_stage.setdefault(r["error_stage"], []).append(r["file"])
        for stage, names in by_stage.items():
            print(f"  {stage}:")
            for n in names:
                print(f"    - {n}")
    print("=" * 60)

    report_path = args.output / "report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report written to {report_path}")
    print(f"Debug dumps under {(args.output / 'mlir')}/<model_name>/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dir",
        type=Path,
        nargs="?",
        help="Directory containing model .py files (e.g. level3)",
    )
    parser.add_argument(
        "--pattern", default=None, help="Only run files whose name matches this regex"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mimir_sweep_results"),
        help="Output directory for debug dumps + report.json",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks for failures"
    )
    parser.add_argument(
        "--allow-graph-breaks",
        action="store_true",
        help="Use fullgraph=False (allow dynamo to silently fall back to eager on "
        "graph breaks). Off by default so a PASS means the whole model actually "
        "went through mimir as one graph.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Don't downgrade PASS->FAIL when the backend logs internal error markers "
        "(e.g. 'unhandled') even if numerics still matched.",
    )
    parser.add_argument(
        "--emit-mlir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the MLIR emitter, writing <name>.mlir next to the .mim/.ll/.so "
        "dumps. Off by default; --no-emit-mlir states that explicitly.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Per-model timeout in seconds (default: 600). A model exceeding this is "
        "killed and marked FAIL(timeout); the sweep continues.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="Cap each model subprocess's virtual memory (POSIX only). A model that "
        "exceeds this gets a clean MemoryError instead of an OOM-kill, so it's "
        "reported as a normal FAIL instead of disappearing silently.",
    )

    # Internal flags used when this script re-invokes itself as a worker subprocess.
    parser.add_argument("--single", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--result-file", type=Path, default=None, help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    if args.single is not None:
        run_child(args)
    else:
        if args.dir is None:
            parser.error("the 'dir' argument is required")
        run_parent(args)


if __name__ == "__main__":
    main()
