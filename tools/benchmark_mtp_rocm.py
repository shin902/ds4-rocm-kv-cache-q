#!/usr/bin/env python3
"""Reproducible DeepSeek V4 Flash ROCm MTP benchmark.

Runs the target-only baseline and the legacy MTP path with draft depths 1, 2,
and 3 using one fixed prompt and fixed inference flags.  Raw stdout/stderr are
kept beside a machine-readable summary so acceptance and timing numbers can be
checked without rerunning the model.

The default paths match the local benchmark machine.  Override them when
running elsewhere, for example:

  tools/benchmark_mtp_rocm.py --target MODEL.gguf --draft MTP.gguf \
      --binary ./ds4 --out-dir results/mtp --repeats 3 --hash-models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_TARGET = (
    "/home/shi/models/DeepSeek-V4-Flash-0731/"
    "DeepSeek-V4-Flash-0731-K160-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
)
DEFAULT_DRAFT = (
    "/home/shi/models/DeepSeek-V4-Flash-0731/"
    "DeepSeek-V4-Flash-0731-K160-REAP-MTP-Q8_0-ROCm.gguf"
)
DEFAULT_PROMPT = (
    "Explain the purpose of speculative decoding in one concise paragraph, "
    "including acceptance rate and verification cost."
)
TIMING_RE = re.compile(
    r"ds4: mtp timing (?P<mode>\S+) drafted=(?P<drafted>\d+) "
    r"committed=(?P<committed>\d+)"
)
NUMBER_RE = re.compile(r"(?P<name>draft|verify|replay|total)=([0-9]+(?:\.[0-9]+)?) ms")
TPS_RE = re.compile(
    r"ds4: prefill: (?P<prefill>[0-9]+(?:\.[0-9]+)?) t/s, "
    r"generation: (?P<generation>[0-9]+(?:\.[0-9]+)?) t/s"
)
MISS_RE = re.compile(r"ds4: mtp spec miss first")
PREPARE_FAIL_RE = re.compile(r"ds4: mtp probe draft failed")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def file_identity(path: Path, hash_files: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_files:
        result["sha256"] = sha256_file(path)
    return result


def stale_ds4_lock(lock_path: Path = Path("/tmp/ds4.lock")) -> None:
    """Remove only a lock whose recorded process no longer exists."""
    if not lock_path.exists():
        return
    try:
        pid = int(lock_path.read_text(encoding="ascii").strip())
        os.kill(pid, 0)
    except ProcessLookupError:
        lock_path.unlink(missing_ok=True)
    except (ValueError, PermissionError):
        raise RuntimeError(f"refusing to remove unreadable active lock: {lock_path}")
    except OSError as exc:
        if exc.errno == 3:  # ESRCH, for platforms where ProcessLookupError is not used
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"cannot verify ds4 lock {lock_path}: {exc}") from exc
    else:
        raise RuntimeError(f"another ds4 process is active (pid {pid})")


def parse_timing(stderr: str, configured_draft: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue
        row: dict[str, Any] = {
            "mode": match.group("mode"),
            "drafted": int(match.group("drafted")),
            "committed": int(match.group("committed")),
        }
        for number in NUMBER_RE.finditer(line):
            row[f"{number.group('name')}_ms"] = float(number.group(2))
        rows.append(row)

    first_misses = sum(1 for line in stderr.splitlines() if MISS_RE.search(line))
    prepare_failures = sum(1 for line in stderr.splitlines() if PREPARE_FAIL_RE.search(line))
    attempts = first_misses + prepare_failures + len(rows)
    proposed = first_misses * configured_draft + sum(r["drafted"] for r in rows)
    accepted = sum(r["committed"] for r in rows)

    def average(name: str, selected: list[dict[str, Any]] | None = None) -> float | None:
        values = [r[name] for r in (selected or rows) if name in r]
        return mean(values) if values else None

    by_depth: dict[str, Any] = {}
    for depth in sorted({r["drafted"] for r in rows}):
        selected = [r for r in rows if r["drafted"] == depth]
        by_depth[str(depth)] = {
            "cycles": len(selected),
            "draft_ms_mean": average("draft_ms", selected),
            "verify_ms_mean": average("verify_ms", selected),
            "replay_ms_mean": average("replay_ms", selected),
            "total_ms_mean": average("total_ms", selected),
        }

    return {
        "configured_draft": configured_draft,
        "first_misses": first_misses,
        "prepare_failures": prepare_failures,
        "timed_cycles": len(rows),
        "attempted_cycles": attempts,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else None,
        "accepted_tokens_per_cycle": accepted / attempts if attempts else None,
        "timing_rows": rows,
        "timing_by_drafted_depth": by_depth,
        "draft_ms_mean": average("draft_ms"),
        "verify_ms_mean": average("verify_ms"),
        "replay_ms_mean": average("replay_ms"),
        "total_ms_mean": average("total_ms"),
        "modes": {
            mode: sum(1 for row in rows if row["mode"] == mode)
            for mode in sorted({row["mode"] for row in rows})
        },
        "committed_histogram": {
            str(committed): sum(1 for row in rows if row["committed"] == committed)
            for committed in sorted({row["committed"] for row in rows})
        },
    }


def parse_run(stdout: str, stderr: str, configured_draft: int | None) -> dict[str, Any]:
    tps_matches = list(TPS_RE.finditer(stderr))
    if not tps_matches:
        raise RuntimeError("could not find the final prefill/generation timing line")
    tps = tps_matches[-1]
    result: dict[str, Any] = {
        "prefill_tps": float(tps.group("prefill")),
        "generation_tps": float(tps.group("generation")),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stdout_bytes": len(stdout.encode("utf-8")),
    }
    if configured_draft is not None and configured_draft > 1:
        result["mtp"] = parse_timing(stderr, configured_draft)
    else:
        result["mtp"] = None
    return result


def command_for(args: argparse.Namespace, draft: int | None) -> list[str]:
    command = [
        str(Path(args.binary).resolve()),
        "--rocm",
        "-m",
        str(Path(args.target)),
        "--ctx",
        str(args.context),
        "--kv-cache-tq4",
        "--temp",
        "0",
        "--nothink",
        "-n",
        str(args.tokens),
        "-p",
        args.prompt,
    ]
    if draft is not None:
        command.extend(["--mtp", str(Path(args.draft).resolve()), "--mtp-draft", str(draft)])
    return command


def run_one(args: argparse.Namespace, label: str, draft: int | None, repeat: int,
            output_dir: Path) -> dict[str, Any]:
    stale_ds4_lock()
    command = command_for(args, draft)
    environment = os.environ.copy()
    environment.update({
        "DS4_CUDA_NO_Q8_F16_CACHE": "1",
        "DS4_MTP_MIN_MARGIN": str(args.mtp_margin),
        "DS4_MTP_TIMING": "1",
        "DS4_MTP_SPEC_LOG": "1",
        "LC_ALL": "C",
    })
    # The engine normally stays silent when a draft graph cannot be prepared.
    # Enable its diagnostic line only for speculative depths; this makes an
    # unsupported draft quantization visible without changing inference.
    if draft is not None and draft > 1:
        environment["DS4_MTP_PROBE"] = "1"
    stdout_path = output_dir / f"{label}-r{repeat}.stdout.log"
    stderr_path = output_dir / f"{label}-r{repeat}.stderr.log"
    completed = subprocess.run(
        command,
        cwd=Path(args.binary).resolve().parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} repeat {repeat} failed with exit code {completed.returncode}; "
            f"see {stderr_path}"
        )
    parsed = parse_run(completed.stdout, completed.stderr, draft if draft and draft > 1 else None)
    parsed.update({
        "label": label,
        "repeat": repeat,
        "draft": draft,
        "command": command,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    })
    return parsed


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    generations = [run["generation_tps"] for run in runs]
    prefills = [run["prefill_tps"] for run in runs]
    hashes = {run["stdout_sha256"] for run in runs}
    result: dict[str, Any] = {
        "runs": len(runs),
        "generation_tps_mean": mean(generations),
        "generation_tps_median": median(generations),
        "prefill_tps_mean": mean(prefills),
        "output_deterministic": len(hashes) == 1,
        "output_sha256": sorted(hashes),
    }
    mtp_runs = [run["mtp"] for run in runs if run["mtp"] is not None]
    if mtp_runs:
        draft_values = [r["draft_ms_mean"] for r in mtp_runs if r["draft_ms_mean"] is not None]
        verify_values = [r["verify_ms_mean"] for r in mtp_runs if r["verify_ms_mean"] is not None]
        proposed = sum(r["proposed_tokens"] for r in mtp_runs)
        attempts = sum(r["attempted_cycles"] for r in mtp_runs)
        accepted = sum(r["accepted_tokens"] for r in mtp_runs)
        result["mtp"] = {
            "attempted_cycles": attempts,
            "prepare_failures": sum(r["prepare_failures"] for r in mtp_runs),
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "accepted_tokens_per_cycle": accepted / attempts if attempts else None,
            "draft_ms_mean": mean(draft_values) if draft_values else None,
            "verify_ms_mean": mean(verify_values) if verify_values else None,
        }
    else:
        result["mtp"] = None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./ds4")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--out-dir", default="bench-results/mtp-rocm-q8_0")
    parser.add_argument("--context", type=int, default=100000)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mtp-margin", type=float, default=3.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--hash-models", action="store_true",
                        help="also record SHA-256 for the large GGUF files")
    args = parser.parse_args(argv)

    if args.repeats < 1 or args.tokens < 1 or args.context < 2:
        parser.error("--repeats, --tokens, and --context must be positive")
    binary = Path(args.binary).resolve()
    target = Path(args.target).resolve()
    draft = Path(args.draft).resolve()
    for path in (binary, target, draft):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "benchmark": "deepseek4_flash_rocm_mtp",
        "binary": file_identity(binary, True),
        "target": file_identity(target, args.hash_models),
        "draft": file_identity(draft, args.hash_models),
        "flags": {
            "backend": "rocm",
            "context": args.context,
            "kv_cache": "tq4",
            "temperature": 0,
            "seed": None,
            "nothink": True,
            "tokens": args.tokens,
            "mtp_margin": args.mtp_margin,
            "environment": {
                "DS4_CUDA_NO_Q8_F16_CACHE": "1",
                "DS4_MTP_TIMING": "1",
                "DS4_MTP_SPEC_LOG": "1",
                "DS4_MTP_MIN_MARGIN": str(args.mtp_margin),
                "DS4_MTP_PROBE": "1 for draft depths > 1 only (diagnostic)",
                "LC_ALL": "C",
            },
        },
        "prompt": args.prompt,
        "repeats": args.repeats,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    variants: list[tuple[str, int | None]] = [("baseline", None), ("draft1", 1), ("draft2", 2), ("draft3", 3)]
    all_runs: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    try:
        for label, draft_depth in variants:
            runs = [run_one(args, label, draft_depth, repeat, output_dir)
                    for repeat in range(1, args.repeats + 1)]
            all_runs.extend(runs)
            aggregates[label] = aggregate(runs)
    except (OSError, RuntimeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1

    baseline_tps = aggregates["baseline"]["generation_tps_median"]
    for label, summary in aggregates.items():
        summary["generation_vs_baseline_percent"] = (
            100.0 * (summary["generation_tps_median"] / baseline_tps - 1.0)
        )

    result = {"manifest": manifest, "aggregates": aggregates, "runs": all_runs}
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("variant\tprefill_tps\tgeneration_tps\tvs_baseline\tacceptance\taccepted/cycle\tdraft_ms\tverify_ms\tdeterministic")
    for label, _ in variants:
        summary = aggregates[label]
        mtp = summary["mtp"]
        acceptance = (
            f"{100.0 * mtp['acceptance_rate']:.2f}%"
            if mtp and mtp["acceptance_rate"] is not None else "N/A"
        )
        accepted_cycle = (
            f"{mtp['accepted_tokens_per_cycle']:.3f}"
            if mtp and mtp["accepted_tokens_per_cycle"] is not None else "N/A"
        )
        draft_ms = (
            f"{mtp['draft_ms_mean']:.3f}"
            if mtp and mtp["draft_ms_mean"] is not None else "N/A"
        )
        verify_ms = (
            f"{mtp['verify_ms_mean']:.3f}"
            if mtp and mtp["verify_ms_mean"] is not None else "N/A"
        )
        print(
            f"{label}\t{summary['prefill_tps_mean']:.2f}\t"
            f"{summary['generation_tps_median']:.2f}\t"
            f"{summary['generation_vs_baseline_percent']:+.2f}%\t"
            f"{acceptance}\t{accepted_cycle}\t{draft_ms}\t{verify_ms}\t"
            f"{str(summary['output_deterministic']).lower()}"
        )
    print(f"raw logs and summary: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
