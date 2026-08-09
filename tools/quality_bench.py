#!/usr/bin/env python3
"""Persistent-server A/B quality and performance benchmark for ds4-server.

Each quantization/topology/implementation variant starts one server and then
receives every selected case.  This deliberately keeps model loading outside
the measurement loop.  Results are JSON and separate quality (output/hash)
from performance (latency/throughput/cache) measurements.

The script uses only the Python standard library and never touches a process
it did not start.  It is therefore safe to use while another ds4-server is
running, provided that ``--port`` is different.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUANT_FLAGS = {
    "f16": (),
    "fp8": ("--kv-cache-fp8",),
    "q8": ("--kv-cache-q8",),
    "tq4": ("--kv-cache-tq4",),
    "tq2": ("--kv-cache-tq2",),
}

RESOURCE_PRESSURE_PHRASES = (
    "cache budget exhausted",
    "out of memory",
    "allocation failed",
    "memory allocation failed",
    "hip out of memory",
    "hiperroroutofmemory",
    "cuda out of memory",
)


def parse_meminfo(text: str) -> dict[str, int] | None:
    """Parse Linux ``/proc/meminfo`` values and report MiB totals.

    The kernel normally reports kB, but accepting the other documented units
    keeps this helper deterministic and useful with fixture data on any host.
    Missing or malformed total/available fields make the sample unavailable.
    """
    unit_scale = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        match = re.match(r"\s*(\d+)\s*([A-Za-z]*)", raw)
        if not match:
            continue
        scale = unit_scale.get(match.group(2).lower(), 1)
        values[key.strip()] = int(match.group(1)) * scale
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None or total < 0 or available < 0:
        return None
    return {
        "total_mib": total // (1024**2),
        "available_mib": available // (1024**2),
        "used_mib": max(0, total - available) // (1024**2),
    }


def read_system_ram() -> dict[str, Any] | None:
    """Best-effort system RAM telemetry; Linux only by design."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text(encoding="ascii"))
    except (OSError, UnicodeError):
        return None


def select_restore_case(cases: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Select the strongest durable-restore probe, preferring long then short."""
    for preferred in ("long", "short"):
        for case in cases:
            if case[0] == preferred:
                return case
    return cases[0] if cases else None


def extract_resource_pressure_hints(log_text: str, max_matches: int = 16) -> dict[str, Any]:
    """Extract bounded resource-pressure hints without embedding server logs."""
    if max_matches <= 0:
        return {"matched_phrases": [], "match_count": 0}
    matched: list[str] = []
    lines = 0
    lowered_phrases = tuple((phrase, phrase.lower()) for phrase in RESOURCE_PRESSURE_PHRASES)
    for line in log_text.splitlines():
        line_lower = line.lower()
        for phrase, needle in lowered_phrases:
            if needle in line_lower:
                lines += 1
                if phrase not in matched:
                    matched.append(phrase)
                if lines >= max_matches:
                    break
        if lines >= max_matches:
            break
    return {"matched_phrases": matched, "match_count": lines}


def read_resource_pressure_hints(paths: list[Path], max_bytes: int = 1 << 20) -> dict[str, Any]:
    """Scan at most ``max_bytes`` from each log and retain only phrase names."""
    text_parts: list[str] = []
    for path in paths:
        try:
            with path.open("rb") as stream:
                text_parts.append(stream.read(max_bytes).decode("utf-8", "replace"))
        except OSError:
            continue
    return extract_resource_pressure_hints("\n".join(text_parts))


@dataclass
class RequestResult:
    status: str
    elapsed_s: float
    http_status: int | None = None
    error: str | None = None
    text: str = ""
    usage: dict[str, Any] | None = None


class VramSampler:
    """Best-effort VRAM sampler; unavailable telemetry never fails a run."""

    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.ram_samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def read() -> dict[str, Any] | None:
        commands = [
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            ["amd-smi", "monitor", "--json"],
            ["rocm-smi", "--showmeminfo", "vram"],
        ]
        for cmd in commands:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
            except (OSError, subprocess.SubprocessError):
                continue
            if p.returncode != 0:
                continue
            values = []
            rocm_total: list[int] = []
            rocm_used: list[int] = []
            for line in p.stdout.splitlines():
                if "VRAM Total Memory" in line:
                    nums = re.findall(r"([0-9]+)\s*$", line)
                    if nums:
                        rocm_total.append(int(nums[-1]))
                    continue
                if "VRAM Total Used Memory" in line:
                    nums = re.findall(r"([0-9]+)\s*$", line)
                    if nums:
                        rocm_used.append(int(nums[-1]))
                    continue
                # nvidia-smi emits ``used, total`` in MiB; ROCm emits several
                # formats, so accepting all integer pairs is intentionally loose.
                nums = [int(x) for x in re.findall(r"(?<![A-Za-z])([0-9]+)", line)]
                if len(nums) >= 2 and nums[0] <= nums[1] * 2:
                    values.append({"used_mib": nums[0], "total_mib": nums[1]})
            if rocm_total and rocm_used:
                values.extend({"used_mib": round(used / (1024 * 1024)),
                               "total_mib": round(total / (1024 * 1024))}
                              for used, total in zip(rocm_used, rocm_total) if used <= total)
            if values:
                return {"source": cmd[0], "devices": values}
        return None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ram = read_system_ram()
            if ram:
                ram["time_unix"] = time.time()
                self.ram_samples.append(ram)
            sample = self.read()
            if sample:
                sample["time_unix"] = time.time()
                self.samples.append(sample)
            self._stop.wait(0.5)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def peak(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        peak = max(max((d["used_mib"] for d in s["devices"]), default=0) for s in self.samples)
        return {"peak_used_mib": peak, "samples": len(self.samples), "source": self.samples[-1]["source"]}

    def system_ram_peak(self) -> dict[str, Any] | None:
        if not self.ram_samples:
            return None
        total = max((s.get("total_mib", 0) for s in self.ram_samples), default=0)
        minimum_available = min((s["available_mib"] for s in self.ram_samples), default=0)
        maximum_used = max((s["used_mib"] for s in self.ram_samples), default=0)
        return {
            "total_mib": total,
            "minimum_available_mib": minimum_available,
            "maximum_used_mib": maximum_used,
            "samples": len(self.ram_samples),
            "source": "/proc/meminfo",
        }


class Server:
    def __init__(self, cmd: list[str], host: str, port: int, log_path: Path, wait_s: float) -> None:
        self.cmd, self.host, self.port = cmd, host, port
        self.log_path, self.wait_s = log_path, wait_s
        self.proc: subprocess.Popen[str] | None = None
        self.log_file = None
        self.sampler = VramSampler()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        self.log_file = log
        self.proc = subprocess.Popen(
            self.cmd, stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True, env=clean_quant_env(),
        )
        self.sampler.start()
        deadline = time.monotonic() + self.wait_s
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                last_error = f"server exited during startup (code {self.proc.returncode})"
                break
            try:
                status, _ = http_json("GET", self.url + "/v1/models", None, 2)
                if status == 200:
                    return
                last_error = f"/v1/models returned HTTP {status}"
            except Exception as exc:  # startup races are expected
                last_error = str(exc)
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(last_error)

    def stop(self) -> int | None:
        self.sampler.stop()
        if not self.proc:
            return None
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.proc.poll() is None:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5)
        code = self.proc.returncode
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        return code


def clean_quant_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("DS4_KV_CACHE_FP8", "DS4_KV_CACHE_Q8", "DS4_KV_CACHE_TQ4", "DS4_KV_CACHE_TQ2"):
        env.pop(key, None)
    return env


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = {"error": raw}
        return exc.code, obj


def request(server: Server, prompt: str, seed: int, max_tokens: int, timeout: float) -> RequestResult:
    payload = {
        "model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "seed": seed,
        "stream": False, "thinking": {"type": "disabled"},
    }
    started = time.monotonic()
    try:
        status, obj = http_json("POST", server.url + "/v1/chat/completions", payload, timeout)
    except Exception as exc:
        return RequestResult("error", time.monotonic() - started, error=str(exc))
    elapsed = time.monotonic() - started
    if status < 200 or status >= 300:
        return RequestResult("error", elapsed, http_status=status, error=str(obj.get("error", obj)))
    choices = obj.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("content") or ""
    return RequestResult("ok", elapsed, status, text=text, usage=obj.get("usage") or {})


def prompt_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    base = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    cases = [("short", base)]
    if args.long_prompt_file:
        cases.append(("long", Path(args.long_prompt_file).read_text(encoding="utf-8")))
    elif args.long_repeat:
        cases.append(("long", (base + "\n") * args.long_repeat))
    return cases


def build_cmd(args: argparse.Namespace, binary: str, mode: str, topology: str, port: int, kv_dir: Path | None) -> list[str]:
    cmd = [binary, "--model", args.model, "--host", args.host, "--port", str(port),
           "--ctx", str(args.context), "--tokens", str(args.max_tokens),
           "--prefill-chunk", str(args.prefill_chunk)]
    if args.backend:
        cmd += ["--backend", args.backend]
    if topology == "tp":
        cmd += ["--cuda-tensor-parallel"]
    if args.gpu_devices:
        cmd += ["--gpu-devices", args.gpu_devices]
    if args.gpu_vram:
        cmd += ["--gpu-vram", args.gpu_vram]
    cmd += list(QUANT_FLAGS[mode])
    if kv_dir:
        # Lower the threshold so short deterministic probes also produce a
        # durable checkpoint; callers can override it with --server-arg.
        cmd += ["--kv-disk-dir", str(kv_dir), "--kv-disk-space-mb", str(args.kv_disk_mb),
                "--kv-cache-min-tokens", "1"]
    for extra in args.server_arg:
        cmd.append(extra)
    return cmd


def output_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_output(actual: str, expected: str, scope: str) -> dict[str, Any]:
    """Return explicit, comparable quality fields without hiding near-misses."""
    return {
        "baseline_scope": scope,
        "exact_match": actual == expected,
        "similarity": difflib.SequenceMatcher(None, actual, expected).ratio(),
        "baseline_sha256": output_fingerprint(expected),
    }


def run_variant(args: argparse.Namespace, variant: str, binary: str, mode: str, topology: str,
                cases: list[tuple[str, str]], baseline: dict[tuple[str, str, str], str], out_dir: Path,
                port: int) -> list[dict[str, Any]]:
    kv_dir = (out_dir / "kv" / variant / topology / mode) if args.session_restore else None
    server = Server(build_cmd(args, binary, mode, topology, port, kv_dir), args.host, port,
                    out_dir / "logs" / f"{variant}-{topology}-{mode}.log", args.startup_timeout)
    rows: list[dict[str, Any]] = []
    session_before: dict[str, str] = {}
    vram_phases: dict[str, dict[str, Any] | None] = {}
    ram_phases: dict[str, dict[str, Any] | None] = {}
    phase_exit_codes: dict[str, int | None] = {}
    log_paths: list[Path] = [server.log_path]
    try:
        server.start()
        for case_name, prompt in cases:
            # A one-token request fills the prefix; the measured request then
            # exercises decode while retaining a live session cache.
            warm = request(server, prompt, args.seed, 1, args.request_timeout)
            measured = request(server, prompt, args.seed, args.max_tokens, args.request_timeout)
            if server.proc is not None and server.proc.poll() is not None:
                # A process can die after accepting a request; retain the
                # response but expose the lifecycle failure distinctly.
                measured.status = "crash"
                measured.error = measured.error or f"server exited (code {server.proc.returncode})"
            usage = measured.usage or {}
            text = measured.text
            key = (variant, topology, case_name)
            quality: dict[str, Any] = {
                "output_sha256": output_fingerprint(text),
                "output_chars": len(text),
                "output": text[: args.output_max_chars],
                "output_truncated": len(text) > args.output_max_chars,
            }
            base = baseline.get((variant, topology, case_name))
            if mode == "f16" and measured.status == "ok":
                baseline[key] = text
                if variant != "integrated":
                    integrated = baseline.get(("integrated", topology, case_name))
                    if integrated is not None:
                        quality.update(compare_output(text, integrated, "integrated/f16"))
            elif base is not None:
                quality.update(compare_output(text, base, "same-variant/f16"))
            if measured.status == "ok":
                session_before[case_name] = text
            row = {"variant": variant, "topology": topology, "mode": mode, "case": case_name,
                   "status": measured.status, "error": measured.error,
                   "quality": quality, "performance": {
                       "elapsed_s": measured.elapsed_s,
                       "prefill_probe_elapsed_s": warm.elapsed_s,
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens"),
                       "estimated_decode_seconds_at_baseline": args.max_tokens / args.decode_baseline_tps,
                       "decode_baseline_tps": args.decode_baseline_tps,
                       "prompt_tokens_per_total_second": rate(usage.get("prompt_tokens"), measured.elapsed_s),
                       "decode_tokens_per_total_second": rate(usage.get("completion_tokens"), measured.elapsed_s),
                       "decode_speed_ratio_vs_baseline": (
                           rate(usage.get("completion_tokens"), measured.elapsed_s) / args.decode_baseline_tps
                           if rate(usage.get("completion_tokens"), measured.elapsed_s) is not None else None),
                       "timing_note": "Both rates divide tokens by total HTTP elapsed (prefill + decode); they are estimates.",
                       "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens")
                           if isinstance(usage.get("prompt_tokens_details"), dict) else None,
                       "warmup_status": warm.status,
                   }}
            rows.append(row)
        if args.session_restore and cases:
            # Stop/restart is the durable save/restore check.  The same server
            # command and KV directory make the prompt hit the disk checkpoint.
            code = server.stop()
            vram_phases["initial"] = server.sampler.peak()
            ram_phases["initial"] = server.sampler.system_ram_peak()
            phase_exit_codes["initial"] = code
            server = Server(build_cmd(args, binary, mode, topology, port, kv_dir), args.host, port,
                            out_dir / "logs" / f"{variant}-{topology}-{mode}-restore.log", args.startup_timeout)
            log_paths.append(server.log_path)
            server.start()
            selected_case = select_restore_case(cases)
            if selected_case is None:
                raise RuntimeError("session restore requested without any prompt cases")
            name, prompt = selected_case
            restored = request(server, prompt, args.seed, args.max_tokens, args.request_timeout)
            expected = session_before.get(name)
            if restored.status == "ok" and expected is not None:
                restore_quality = compare_output(restored.text, expected, "same-process-before-restart")
                restore_quality["session_before_sha256"] = output_fingerprint(expected)
            else:
                restore_quality = {
                    "baseline_scope": "same-process-before-restart",
                    "exact_match": None,
                    "similarity": None,
                    "session_before_sha256": output_fingerprint(expected) if expected is not None else None,
                }
            rows.append({"variant": variant, "topology": topology, "mode": mode, "case": "session_restore",
                         "restored_case": name,
                         "status": restored.status, "error": restored.error,
                         "quality": {**restore_quality, "output_sha256": output_fingerprint(restored.text)},
                         "performance": {"elapsed_s": restored.elapsed_s, "usage": restored.usage,
                                         "prior_exit_code": code}})
    except Exception as exc:
        rows.append({"variant": variant, "topology": topology, "mode": mode, "case": "server",
                     "status": "crash", "error": str(exc), "quality": {}, "performance": {}})
    finally:
        exit_code = server.stop()
        phase = "restore" if args.session_restore and "initial" in vram_phases else "initial"
        vram_phases[phase] = server.sampler.peak()
        ram_phases[phase] = server.sampler.system_ram_peak()
        phase_exit_codes[phase] = exit_code
        resource_hints = read_resource_pressure_hints(log_paths)
        for row in rows:
            diagnostics = row.setdefault("diagnostics", {})
            diagnostics["server_exit_code"] = exit_code
            diagnostics["phase_exit_codes"] = phase_exit_codes
            diagnostics["vram"] = vram_phases
            diagnostics["system_ram"] = ram_phases
            diagnostics["resource_pressure_hints"] = resource_hints
            peaks = [v.get("peak_used_mib") for v in vram_phases.values() if v and v.get("peak_used_mib") is not None]
            diagnostics["vram_peak_used_mib"] = max(peaks) if peaks else None
    return rows


def rate(value: Any, seconds: float) -> float | None:
    try:
        return float(value) / seconds if value is not None and seconds > 0 else None
    except (TypeError, ValueError):
        return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--binary", default="./ds4-server", help="integrated server binary")
    p.add_argument("--upstream-binary", help="optional upstream/reference server binary")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file")
    p.add_argument("--prompt", default="Reply with exactly one short sentence describing a cache benchmark.")
    p.add_argument("--long-prompt-file")
    p.add_argument("--long-repeat", type=int, default=0, help="repeat base prompt to create a long case")
    p.add_argument("--modes", default="f16,fp8,q8,tq4,tq2", help="comma-separated KV modes")
    p.add_argument("--topology", choices=("single", "tp", "both"), default="single")
    p.add_argument("--backend", choices=("cpu", "cuda", "rocm", "metal"))
    p.add_argument("--gpu-devices")
    p.add_argument("--gpu-vram")
    p.add_argument("--context", type=int, default=32768, help="conservative context; increase only after checking VRAM")
    p.add_argument("--prefill-chunk", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=32, help="decode tokens; ~15 TPS implies a few seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--startup-timeout", type=float, default=180)
    p.add_argument("--request-timeout", type=float, default=300)
    p.add_argument("--decode-baseline-tps", type=float, default=15.0,
                   help="reference decode speed used for duration estimates, not a pass/fail gate")
    p.add_argument("--output", default="quality-bench.json")
    p.add_argument("--output-max-chars", type=int, default=4096)
    p.add_argument("--server-arg", action="append", default=[], help="extra server argument (repeatable, one token each)")
    p.add_argument("--session-restore", action="store_true", help="restart each server and verify durable KV restore")
    p.add_argument("--kv-disk-mb", type=int, default=2048)
    p.add_argument("--dry-run", action="store_true", help="print planned server commands without starting models")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    invalid = sorted(set(modes) - set(QUANT_FLAGS))
    if invalid:
        raise SystemExit(f"invalid --modes: {', '.join(invalid)}")
    # Quality deltas are relative to F16.  Keep it first even when a caller
    # supplies modes in a different order; modes without F16 remain valid
    # performance-only runs.
    if "f16" in modes:
        modes = ["f16"] + [m for m in modes if m != "f16"]
    if args.max_tokens <= 0 or args.context <= 0 or args.prefill_chunk <= 0 or args.decode_baseline_tps <= 0:
        raise SystemExit("--context, --prefill-chunk, --max-tokens and --decode-baseline-tps must be positive")
    binaries = [("integrated", args.binary)]
    if args.upstream_binary:
        binaries.append(("upstream", args.upstream_binary))
    # The reference process is intentionally F16-only.  Quantized cache modes
    # belong to the integrated implementation and would multiply model loads
    # without answering the upstream-vs-integrated smoke question.
    variant_modes = {"integrated": modes, "upstream": ["f16"]}
    topologies = ("single", "tp") if args.topology == "both" else (args.topology,)
    cases = prompt_cases(args)
    out_path = Path(args.output)
    out_dir = out_path.parent / (out_path.stem + ".artifacts")
    plans = []
    port = args.port
    for variant, binary in binaries:
        for topology in topologies:
            for mode in variant_modes[variant]:
                planned_kv = (out_dir / "kv" / variant / topology / mode) if args.session_restore else None
                plans.append({"variant": variant, "topology": topology, "mode": mode,
                              "command": build_cmd(args, binary, mode, topology, port, planned_kv)})
                port += 1
    if args.dry_run:
        print(json.dumps({"cases": [name for name, _ in cases], "plans": plans}, indent=2))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline: dict[tuple[str, str, str], str] = {}
    rows: list[dict[str, Any]] = []
    port = args.port
    for variant, binary in binaries:
        for topology in topologies:
            for mode in variant_modes[variant]:
                rows.extend(run_variant(args, variant, binary, mode, topology, cases, baseline, out_dir, port))
                port += 1
    result = {"schema": "ds4-quality-bench/v1", "created_unix": time.time(),
              "config": {k: v for k, v in vars(args).items() if k not in ("prompt",)},
              "cases": [{"name": n, "prompt_sha256": output_fingerprint(p)} for n, p in cases],
              "results": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} result rows); logs/artifacts: {out_dir}")
    return 0 if all(r.get("status") == "ok" for r in rows if r.get("case") != "server") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
