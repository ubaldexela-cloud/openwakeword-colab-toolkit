#!/usr/bin/env python3
"""Checks that can run on the HPC without training or a Pi."""
from __future__ import annotations

import ast
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestIcyConfig(unittest.TestCase):
    def test_yaml_name_and_phrase(self):
        text = (ROOT / "icy.yaml").read_text()
        try:
            import yaml
            cfg = yaml.safe_load(text)
        except ImportError:
            cfg = None
        if cfg is not None:
            self.assertEqual(cfg["model_name"], "icy")
            self.assertEqual(cfg["target_phrase"], ["icy"])
            self.assertEqual(cfg["output_dir"], "./icy_model")
            self.assertGreaterEqual(cfg["n_samples"], 25000)
            self.assertIn("ice", cfg["custom_negative_phrases"])
            self.assertIn("i see", cfg["custom_negative_phrases"])
            self.assertNotIn("icy", cfg["custom_negative_phrases"])
            self.assertEqual(cfg["model_type"], "dnn")
            self.assertEqual(cfg["piper_sample_generator_path"], "./piper-sample-generator")
        else:
            self.assertIn('model_name: "icy"', text)
            self.assertIn('  - "icy"', text.split("target_phrase:", 1)[1].split("custom_negative", 1)[0])
            self.assertIn("n_samples: 50000", text)
            self.assertIn('  - "ice"', text)
            self.assertIn('  - "i see"', text)
            negatives = text.split("custom_negative_phrases:", 1)[1].split("n_samples:", 1)[0]
            self.assertNotIn('  - "icy"\n', negatives)

    def test_notebook_writes_icy_yaml(self):
        nb = json.loads((ROOT / "icy_colab.ipynb").read_text())
        blob = "".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("%%writefile icy.yaml", blob)
        self.assertIn('model_name: "icy"', blob)
        self.assertIn("icy.onnx", blob)
        self.assertIn("icy_outputs.zip", blob)
        self.assertNotIn("hey_clem.yaml", blob)
        self.assertNotIn("hey_clem.onnx", blob)

    def test_scripts_compile(self):
        for name in (
            "oww_train_shim.py",
            "check_model.py",
            "phrase_check.py",
            "slice_positives.py",
            "pi/listen_icy.py",
            "tests/test_icy_config.py",
        ):
            py_compile.compile(str(ROOT / name), doraise=True)

    def test_shim_is_valid_python(self):
        src = (ROOT / "oww_train_shim.py").read_text()
        ast.parse(src)
        self.assertIn('runpy.run_module("openwakeword.train"', src)

    def test_check_model_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "check_model.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--model", r.stdout)
        self.assertIn("--negatives", r.stdout)
        self.assertIn("--positives", r.stdout)

    def test_listen_icy_help_and_missing_model(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "pi/listen_icy.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("inference_framework", (ROOT / "pi/listen_icy.py").read_text())
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "pi/listen_icy.py"),
             "--model", "/no/such/icy.onnx"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r2.returncode, 1)

    def test_count_activations(self):
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import check_model

        scores = np.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.9, 0.1])
        self.assertEqual(check_model.count_activations(scores, 0.5), 2)
        self.assertEqual(check_model.count_activations(np.array([0.1, 0.2]), 0.5), 0)

    def test_slice_positives_splits_two_bursts(self):
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import slice_positives

        rate = 16000
        silence = np.zeros(rate, dtype=np.int16)
        burst = np.full(int(0.4 * rate), 8000, dtype=np.int16)
        audio = np.concatenate([silence, burst, silence, burst, silence])
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "session.wav")
            import wave
            with wave.open(src, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(audio.tobytes())
            out = os.path.join(td, "clips")
            r = subprocess.run(
                [sys.executable, str(ROOT / "slice_positives.py"), src, out, "--label", "icy"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            clips = sorted(Path(out).glob("icy_*.wav"))
            self.assertGreaterEqual(len(clips), 2)


if __name__ == "__main__":
    unittest.main()
