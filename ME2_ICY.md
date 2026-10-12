# ME2 “Icy” wake word — where each step runs

This fork trains an openWakeWord model for the spoken wake word **Icy**
(output name `icy`). The original Clem trainer (`hey_clem_colab.ipynb`,
`hey_clem.yaml`) is left in place as the upstream workflow.

**The model has not been trained or measured in this change.** Do not
claim it works on the Pi until Colab (or an equivalent GPU job) has
exported `icy.onnx` **and** `check_model.py` has been run on real room
audio.

Do not touch:

- `/home/xela.ubalde/tmp_me2_vcm_deploy_check/pi_package_lightweight/`
- `/home/xela.ubalde/tmp_me2_vcm_deploy_check/pi_package_scratch/`
- `~/me2_vcm/pi_package_lightweight/` on the Pi
- `~/me2_vcm/pi_package_scratch/` on the Pi

---

## Who does what

| Step | Where | Notes |
| --- | --- | --- |
| Inspect / edit config, docs, tests | **HPC (here)** | Already done in this repo. |
| `python tests/test_icy_config.py` | **HPC** | No GPU, no Colab, no Pi. |
| `python phrase_check.py "icy" "hey icy" "hey jarvis"` | **HPC or laptop** | Needs `piper-phonemize` / `espeak-ng`. Optional `--durations` needs piper-sample-generator + LibriTTS checkpoint. |
| Download MIT RIRs, AudioSet, FMA, ACAV100M features | **Google Colab** | Hugging Face. The original FINDINGS.md notes some HPC sandboxes cannot reach `huggingface.co`. |
| Generate 50k TTS clips, augment, train, export ONNX | **Google Colab T4** (`icy_colab.ipynb`) | ~1.5–2 h. The login node has no GPU; do not run this on CPU. A PERLAS GPU job can do the same *if* Hugging Face is reachable from the compute node and the Colab notebook’s deps are installed there. That path is not wired in this commit. |
| Fold `icy.onnx` + `icy.onnx.data` into one file | **HPC or laptop** | `onnx.load` / `save_as_external_data=False` (see README). |
| Record ~1 h of room audio with **no** “Icy” | **Raspberry Pi 5** USB mic (`plughw:2,0`, 16 kHz) | False-activation set. |
| Record ~20 real “Icy” takes, slice with `slice_positives.py --label icy` | **Pi or laptop** | Missed-detection set. |
| `python check_model.py --model icy.onnx --negatives room/ --positives icy_clips/` | **Linux or the Pi** | macOS scored near-zero in the original kit. Gate: ≤0.5 false accepts/hour at ≥80% recall. |
| Copy `icy.onnx` to `~/me2_vcm/wakeword/` and run `pi/listen_icy.py` | **Raspberry Pi 5** | After the gate passes. |

No company email, paid wake-word API, or proprietary service is required.
Colab Pro is optional (the original README warns free-tier sessions can die
mid-job). Training data and openWakeWord are open source.

---

## Reproducible training (Colab)

1. Upload `icy_colab.ipynb` to Google Colab.
2. Runtime → T4 GPU → Run all.
3. Confirm cell 2 preflight prints phonemes for `"icy"` and `PREFLIGHT OK`.
4. Wait for `icy_outputs.zip` to download. It must contain `icy.onnx`
   (and `icy.onnx.data` if present) plus `icy.yaml`.
5. A `ModuleNotFoundError: onnx_tf` at the end of the train cell is
   expected: tflite export is skipped on purpose. The Pi uses ONNX.

Equivalent CLI once the Colab environment exists:

```sh
python oww_train_shim.py --training_config icy.yaml --generate_clips
python oww_train_shim.py --training_config icy.yaml --augment_clips
python oww_train_shim.py --training_config icy.yaml --train_model
```

Export format: **ONNX**, loaded with

```python
openwakeword.model.Model(wakeword_models=["icy.onnx"], inference_framework="onnx")
```

fed **80 ms** chunks (1280 samples) of **16 kHz mono int16**. That is the
current openWakeWord inference API used by `check_model.py` and
`pi/listen_icy.py`.

---

## Validation tests (false activations and misses)

After you have `icy.onnx` and real recordings:

```sh
python slice_positives.py session.wav icy_positives/ --label icy
python check_model.py --model icy.onnx \
    --negatives room_audio/ \
    --positives icy_positives/ \
    --budget 0.5
```

- **False activations:** `--negatives` must be ≥1 hour of ordinary room
  audio with nobody saying “Icy”. The script refuses a verdict on less
  (exit 3) unless you pass `--allow-short`, which is not a deploy signal.
- **Missed detections:** `--positives` is ~20 held-out clips of you saying
  “Icy” on the Pi mic. Recall at the recommended threshold should be ≥80%.
- Exit 0 = a threshold inside budget; exit 2 = not fit to deploy.

Until those recordings exist, the only tests in this repo are
`tests/test_icy_config.py` (config, compile, CLI, slicer). They do **not**
mean the wake word works.

---

## Transfer to the Raspberry Pi (after a passing gate)

On the laptop or HPC, do **not** unpack into the VCM packages.

```sh
scp icy.onnx icy.yaml pi/listen_icy.py \
    xelaubalde@<PI_IP>:~/me2_vcm/wakeword/
```

On the Pi:

```sh
mkdir -p ~/me2_vcm/wakeword
python3 -m pip install --no-deps openwakeword
python3 -m pip install "onnxruntime==1.30.0" numpy tqdm requests scikit-learn scipy
# Same aarch64 wheel rule as the VCM package. Do not pip-install tflite-runtime.

python3 ~/me2_vcm/wakeword/listen_icy.py \
    --model ~/me2_vcm/wakeword/icy.onnx \
    --threshold <value from check_model.py> \
    --device plughw:2,0
```

Say “Icy”. Then say command phrases the VCM already uses (`turn on the
light`, `volume down`, `hello`) and confirm the wake listener stays quiet.
That live check is still required; this document cannot perform it from the
HPC (the cluster cannot SSH to `192.168.100.12`).

---

## Short-phrase risk

`phrase_check.py` and FINDINGS.md both say short names false-accept more.
“Icy” is two syllables. If the gate fails, the next experiment is a longer
phrase such as “hey icy”, not lowering `target_false_positives_per_hour`.
That change is not in this config because the requested wake word is “Icy”.
