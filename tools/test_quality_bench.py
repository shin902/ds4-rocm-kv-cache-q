#!/usr/bin/env python3
"""Fast, model-free checks for the quality benchmark's deterministic pieces."""

import argparse
import contextlib
import io
import json
import unittest

try:
    from . import quality_bench as qb
except ImportError:  # direct ``python tools/test_quality_bench.py``
    import quality_bench as qb


class QualityBenchTest(unittest.TestCase):
    def test_quant_flags_are_exclusive_and_named(self):
        self.assertEqual(qb.QUANT_FLAGS["f16"], ())
        self.assertEqual(qb.QUANT_FLAGS["tq4"], ("--kv-cache-tq4",))
        self.assertNotEqual(qb.QUANT_FLAGS["q8"], qb.QUANT_FLAGS["tq4"])

    def test_rate_and_fingerprint_are_stable(self):
        self.assertEqual(qb.rate(30, 2), 15.0)
        self.assertIsNone(qb.rate(None, 2))
        self.assertEqual(qb.output_fingerprint("same"), qb.output_fingerprint("same"))
        self.assertNotEqual(qb.output_fingerprint("same"), qb.output_fingerprint("different"))

    def test_build_command_keeps_cache_restore_options(self):
        args = argparse.Namespace(model="m.gguf", host="127.0.0.1", context=32768,
                                  max_tokens=32, prefill_chunk=512, backend=None,
                                  gpu_devices=None, gpu_vram=None, server_arg=[], kv_disk_mb=64)
        cmd = qb.build_cmd(args, "./ds4-server", "tq4", "single", 18080, "/tmp/bench-kv")
        self.assertIn("--kv-cache-tq4", cmd)
        self.assertIn("--kv-cache-min-tokens", cmd)
        self.assertNotIn("--cuda-tensor-parallel", cmd)

    def test_upstream_dry_run_is_f16_only(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(qb.main(["--model", "m.gguf", "--binary", "integrated",
                                      "--upstream-binary", "upstream", "--modes", "f16,tq4",
                                      "--dry-run"]), 0)
        plans = json.loads(output.getvalue())["plans"]
        self.assertEqual([(p["variant"], p["mode"]) for p in plans],
                         [("integrated", "f16"), ("integrated", "tq4"), ("upstream", "f16")])

    def test_compare_output_records_scope_and_similarity(self):
        result = qb.compare_output("abc", "abd", "same-process-before-restart")
        self.assertEqual(result["baseline_scope"], "same-process-before-restart")
        self.assertFalse(result["exact_match"])
        self.assertGreater(result["similarity"], 0.5)

    def test_meminfo_parser_reports_total_available_and_used(self):
        sample = "MemTotal:       1024000 kB\nMemAvailable:    256000 kB\n"
        self.assertEqual(qb.parse_meminfo(sample), {
            "total_mib": 1000, "available_mib": 250, "used_mib": 750,
        })
        self.assertIsNone(qb.parse_meminfo("MemTotal: 1 kB\n"))

    def test_restore_prefers_long_case_then_short(self):
        cases = [("short", "small"), ("long", "large")]
        self.assertEqual(qb.select_restore_case(cases), cases[1])
        self.assertEqual(qb.select_restore_case([cases[0]]), cases[0])
        self.assertIsNone(qb.select_restore_case([]))

    def test_resource_pressure_hints_are_bounded_and_do_not_include_logs(self):
        hints = qb.extract_resource_pressure_hints(
            "cache budget exhausted while growing\nnormal completion\nOUT OF MEMORY\n"
        )
        self.assertEqual(hints["matched_phrases"], ["cache budget exhausted", "out of memory"])
        self.assertEqual(hints["match_count"], 2)
        self.assertNotIn("normal completion", hints)


if __name__ == "__main__":
    unittest.main()
