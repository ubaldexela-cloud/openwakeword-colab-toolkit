# Train your own openWakeWord wake word (2026 edition)

## ME2 fork: wake word “Icy”

This checkout is **ubaldexela-cloud/openwakeword-colab-toolkit**, a fork of
saintpete’s kit, configured for the ME2 Voice Controlled Smart Device.

| File | Role |
| --- | --- |
| `icy.yaml` | Training config: phrase `icy`, model name `icy`, 50k TTS clips, near-miss negatives |
| `icy_colab.ipynb` | Colab trainer (same cells as the Clem notebook, Icy config) |
| `ME2_ICY.md` | Which steps run on HPC vs Colab vs Raspberry Pi 5 |
| `pi/listen_icy.py` | Pi 5 listener using the openWakeWord ONNX API |
| `tests/test_icy_config.py` | HPC checks that do not train a model |

The Clem files (`hey_clem_colab.ipynb`, `hey_clem.yaml`) are the original
workflow and are left unchanged.

**No `icy.onnx` has been trained in this fork yet.** Upload `icy_colab.ipynb`
to Colab (T4 GPU), then measure with `check_model.py` before putting anything
on the Pi. Do not overwrite the ME2 VCM Run 3 or Scratch packages.

---

A working end-to-end kit for training a **custom wake word** ("hey Robby",
"okay Mango", …) with [openWakeWord](https://github.com/dscripka/openWakeWord)
on today's Google Colab — plus, more importantly, the **measurement gate**
that tells you whether the result is safe to deploy.

Why this exists: openWakeWord's official training notebook no longer runs on
current Colab (Python 3.12, torch 2.11, reorganized datasets). This kit
encodes ~10 verified fixes — see the "Live-run addendum" in `FINDINGS.md` for
the full list — and was proven end to end on 2026-08-01: the resulting model
measured **0.00 false accepts/hour over 75 minutes of real living-room audio
at 91% recall** on the target device (a Reachy Mini robot).

It was built for a robot named Clem; everything except one config cell is
name-agnostic.

## What's in the box

| File | What it does |
| --- | --- |
| `icy_colab.ipynb` | **ME2 trainer.** Upload to Colab, Run all (~1.5–2 h on a T4). Phrase and model name are already `icy`. |
| `hey_clem_colab.ipynb` | Original Clem trainer, kept as the upstream workflow. |
| `oww_train_shim.py` | Compat shim for `openwakeword.train` on a modern stack (also written by the notebook itself). |
| `check_model.py` | **The deployment gate.** False-accepts/hour + recall vs threshold, from real audio. Do not skip. |
| `phrase_check.py` | Compare candidate phrases (phonemes, clip length) *before* spending an hour training the wrong one. |
| `slice_positives.py` | Splits "me saying the phrase 20 times" into the per-utterance clips the gate needs. |
| `hey_clem.yaml` | The training config, as reference — the notebook embeds its own copy. |
| `FINDINGS.md` | Findings, phrase research, gate criteria, rollback thinking, and every wall we hit with its fix. |

## How to use it

**1. Pick a good phrase.** Short phrases false-accept more. Aim for 3+
syllables / ~8 phonemes ("hello Robby", not "Rob"). If you have `espeak-ng`
locally, `phrase_check.py` will show you phonemes and (with a Piper
checkout) measured clip lengths. Training several phrasings into one model
works — we trained "hey / okay / hello Clem" together, and note that
`n_samples` in the config is a TOTAL split across phrases (~25k per phrase
is a good density).

**2. Edit one cell.** In the notebook, the `%%writefile hey_clem.yaml` cell
holds the whole config. Change:
- `model_name` — yours
- `target_phrase` — your phrase(s), lowercase
- `custom_negative_phrases` — near-misses for YOUR name (rhymes, one-sound-off
  words; ours had "hey clam", "hello glen"…). This is where short wake words
  win or lose.

Leave the rest alone unless you've read the yaml's comments.

**3. Run it.** Colab → File → Upload notebook → Runtime → **T4 GPU** →
Run all.

> ⚠ **Budget for Colab Pro (US$9.99/month).** The full job is ~2 GPU-hours,
> and free-tier credit is unpredictable — in our runs it expired mid-job and
> stranded a nearly-finished model, forcing a rerun. Pro finished the same
> job comfortably; you can cancel after training. If you do try the free
> tier, be prepared for the session to be reclaimed partway through. Keep the tab open. Cell 1 fails fast if you didn't get a GPU;
cell 2 ends with a preflight so environment problems die in the fast cell,
not 30 minutes into training. At the end your browser downloads a zip with
`<name>.onnx` **and `<name>.onnx.data`** — you need BOTH (the weights live
in the sidecar). To fold them into a single file:

```sh
python -c "import onnx; m = onnx.load('hey_clem.onnx'); onnx.save_model(m, 'hey_clem_single.onnx', save_as_external_data=False)"
```

A `ModuleNotFoundError: onnx_tf` at the very end of the training cell is
**expected and harmless** — it's a tflite conversion attempt that fires
after the ONNX is already written.

**4. MEASURE BEFORE YOU DEPLOY.** This is the part most people skip and the
reason this kit exists. A wake word listens all day; an unmeasured one is a
false-accept machine. Record **at least one hour** of ordinary audio from
the room it will live in (any recorder, any sample rate — phone is fine;
nobody says the wake phrase), plus ~20 real recordings of you saying it
(`slice_positives.py` splits a single say-it-repeatedly session). Then:

```sh
python check_model.py --model your_model.onnx --negatives room_audio/ --positives your_clips/
```

It reports false-accepts/hour and recall per threshold and recommends the
threshold — or refuses, which is also an answer. A reasonable bar:
**≤0.5 false accepts/hour at ≥80% recall.** If it can't hit both, don't
lower the bar — train a longer phrase, or look at openWakeWord's
`custom_verifier_model` (gates detections on your specific voice).

⚠ **Score on Linux or on your deployment device** — in our runs, macOS
scored every model near-zero through the same code path that read 0.89+ on
Linux (unresolved platform quirk). The gate's numbers are only as good as
the machine they ran on; run them where the model will actually live.

**5. Deploy** at the recommended threshold, and keep your old wake word one
config line away for rollback.

## What this kit does — and doesn't — provide

This kit produces and validates the **wake-word model**. It does not give
your app ears: something in your stack must route mic audio through
openWakeWord and act on detections. Where that already exists:

- **Anything that embeds openWakeWord** (Python stacks, Wyoming satellites,
  Home Assistant's wake-word ecosystem) can load the `.onnx` directly. Note
  HA's add-on defaults to tflite models; this kit deliberately produces ONNX.
- **Reachy Mini**: the stock conversation app has no wake-word support today.
  Standby-mode support (opt-in `STANDBY_ON_SLEEP=1`, wake model via
  `WAKE_WORD_MODEL` — pretrained name or a custom `.onnx` from this kit) has
  been proposed upstream:
  [pollen-robotics/reachy_mini_conversation_app **PR #514**](https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/514).

  > **PR #514 status: open, awaiting maintainer review** *(last checked
  > 2026-08-02).* This README will be updated once its fate is decided —
  > merged (use the stock app) or not (use the PR branch as the reference
  > integration).

## Runtime notes for your own stack

Loading the model: `openwakeword.model.Model(wakeword_models=["path/to/your.onnx"], inference_framework="onnx")`,
fed 80 ms chunks (1280 samples) of 16 kHz mono int16. On Python 3.12,
install openwakeword with `--no-deps` plus `onnxruntime scipy scikit-learn
requests tqdm` — a plain `pip install openwakeword` silently downgrades to
a fossil version (no py3.12 tflite-runtime exists).

---

*Built with Claude (2026-08-01) for a Reachy Mini named Clem — who woke
to his own name on the first try. License: [Unlicense](LICENSE) — public
domain; no attribution required, no rights reserved.*
