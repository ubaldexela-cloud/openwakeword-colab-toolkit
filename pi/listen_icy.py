#!/usr/bin/env python3
"""Raspberry Pi 5 listener for the ME2 "Icy" openWakeWord model.

This is inference only. It does not load or replace the VCM Run 3 or Scratch
packages. Keep those at:

  ~/me2_vcm/pi_package_lightweight/
  ~/me2_vcm/pi_package_scratch/

Put the wake-word ONNX here:

  ~/me2_vcm/wakeword/icy.onnx

  python listen_icy.py --model ~/me2_vcm/wakeword/icy.onnx

Needs: openwakeword (onnx path), onnxruntime, numpy, and either sounddevice
or `arecord` (ALSA). Capture is 16 kHz mono int16 in 80 ms chunks (1280
samples), matching openWakeWord's inference API.

The model is NOT claimed to work until check_model.py has been run on real
Pi-mic audio. Default threshold 0.5 is a starting point, not a measured gate.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms


def read_chunks_arecord(device: str):
    cmd = [
        "arecord", "-D", device, "-f", "S16_LE", "-c", "1",
        "-r", str(SAMPLE_RATE), "-t", "raw", "-q",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME * 2)
    try:
        while True:
            buf = proc.stdout.read(FRAME * 2)
            if not buf or len(buf) < FRAME * 2:
                break
            yield buf
    finally:
        proc.terminate()


def read_chunks_sounddevice():
    import numpy as np
    import sounddevice as sd

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME
    ) as stream:
        while True:
            data, overflowed = stream.read(FRAME)
            if overflowed:
                print("audio overflow", file=sys.stderr)
            yield bytes(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/me2_vcm/wakeword/icy.onnx"),
                    help="path to icy.onnx")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="detection threshold (use check_model.py to pick this)")
    ap.add_argument("--device", default=os.environ.get("ALSA_DEVICE", "plughw:2,0"),
                    help="ALSA capture device for arecord (default plughw:2,0)")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        print("Train in Colab first (icy_colab.ipynb). This script does not train.",
              file=sys.stderr)
        return 1

    from openwakeword.model import Model
    import numpy as np

    model = Model(wakeword_models=[args.model], inference_framework="onnx")
    key = next(iter(model.models))
    print(f"loaded {args.model} key={key} framework=onnx")
    print(f"threshold={args.threshold} (unmeasured unless check_model.py was run)")

    if shutil.which("arecord"):
        chunks = read_chunks_arecord(args.device)
        print(f"capture: arecord {args.device} 16 kHz mono S16_LE")
    else:
        chunks = read_chunks_sounddevice()
        print("capture: sounddevice default input 16 kHz mono")

    last_fire = 0.0
    cooldown = 1.5
    try:
        for raw in chunks:
            audio = np.frombuffer(raw, dtype=np.int16)
            scores = model.predict(audio)
            score = float(scores[key])
            now = time.monotonic()
            if score >= args.threshold and (now - last_fire) >= cooldown:
                last_fire = now
                print(f"ICY  score={score:.3f}")
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
