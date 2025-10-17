# LossAnalyzer — Training Loss Curve Analyzer

A lightweight GUI/CLI tool to parse your training logs (JSON/JSONL), visualize loss curves, and compute actionable metrics like recent slope, robust noise (MAD), signal‑to‑noise ratio, exposure %, and early‑stop suggestions. Built with Tkinter + Matplotlib and designed to be forgiving about log formats.

> TL;DR: Paste or load your `.jsonl` logs, click **Analyze**, and get an instant plot plus a metrics table you can copy/export as Markdown.

---

## Features

- **Robust parsing** of JSONL and console paste (accepts single‑quoted dicts, mixed blobs).
- **Smoothed curves** using SciPy’s `uniform_filter1d` when available; otherwise a fast moving‑average fallback.
- **Actionable metrics**:
  - Recent trend **slope** (per 100 steps)
  - **MAD** noise (robust to outliers) and **SNR**
  - **Best loss** and **steps since best**
  - **Exposure %** (epoch fraction) with an **18% floor** hint
  - **Learning‑rate** summary (mode, now/max)
  - **Trend label**, **letter grade**, and a **suggestion** (continue / plateau / stop)
- **Plot annotations**: “ideal zone” band (best..best+0.05), epoch markers, outlier highlights, predicted stop (if trend allows).
- **GUI buttons**: Load JSONL, Copy Metrics, Export Markdown, Analyze Now (plus Ctrl+Enter hotkey).
- **CLI mode** for quick plotting without the GUI.

---

## Install

Python 3.10+ recommended.

```bash
# core deps
pip install matplotlib numpy

# optional (improves smoothing & clipboard UX)
pip install scipy pyperclip
```

> **Linux/macOS**: If Tkinter is missing, install your OS package (e.g., Ubuntu: `sudo apt-get install python3-tk`).  
> **Windows**: Tkinter ships with the standard Python installer.

---

## Quick Start

### GUI

```bash
python LossAnalyzer.py
```
1. **Paste** JSON/JSONL rows into the editor, or click **Load JSONL**.  
2. Press **Analyze Now** (or **Ctrl+Enter**).  
3. Copy/export metrics using the **Copy Metrics** or **Export Markdown** buttons.

### CLI

```bash
python LossAnalyzer.py path/to/log.jsonl
```
Opens a plot directly using Matplotlib (no GUI widgets).

---

## Log Format (Flexible)

Each row ideally looks like a JSON dict with at least `loss` and optionally `epoch`, `learning_rate`:

```json
{"loss": 2.9412, "epoch": 0.11, "learning_rate": 5e-5}
{"loss": 2.9140, "epoch": 0.12, "learning_rate": 5e-5}
{"loss": 2.8803, "epoch": 0.13, "learning_rate": 5e-5}
```

The parser also tries to salvage:
- Single‑quoted dicts (e.g., `{'loss': 1.23}`)
- Blobs where multiple `{...}` objects are jammed together
- Mixed lines, as long as a dict contains a `loss` key

---

## What You’ll See

- **Plot** of raw and smoothed loss over steps
- **Best loss** marker and **ideal zone** (best .. best+0.05)
- **Epoch boundary** hints and **outlier highlights**
- **(If LR available)** a dashed overlay of learning rate
- **Predicted stop** line when the trend suggests an ETA

---

## Metrics Explained

| Metric | Meaning |
|---|---|
| Steps | Number of training steps parsed |
| Current Loss | Final loss value |
| Best Loss | Lowest loss and the step when it occurred |
| Drop (total) | Absolute/percent improvement from first to last |
| Slope (last N) | Recent trend (per 100 steps). Negative is better |
| Noise (MAD) | Robust estimate of fluctuations (less is cleaner) |
| Signal/Noise | `abs(slope) / noise` – higher means clearer trend |
| Steps since best | Recency of improvement |
| Exposure | Epoch fraction ×100 (aim for ≥ 18% before stopping) |
| LR, LR now / max | Bucketed LR summary and current vs max LR |
| Trend | Human‑readable label (Strong/Moderate/Weak Drop/Rise, Flat) |
| Grade | A–D score from slope, SNR, exposure |
| Suggestion | “Continue”, “Plateau—consider stop”, “Loss rising—check LR/data”, etc. |

> The **18% exposure floor** helps avoid stopping before the model has seen enough data. Many pipelines benefit from this conservative guardrail.

---

## Buttons & Shortcuts

- **Load JSONL** — pick a file to load into the editor.
- **Analyze Now** — parse & plot the current buffer.
- **Copy Metrics** — copies a plain‑text table; falls back by appending to the editor if clipboard package is missing.
- **Export Markdown** — saves a `| Metric | Value |` table to `.md`.
- **Ctrl+Enter** — quick analyze.

---

## Tips

- If your logs lack `epoch`, exposure is estimated from what’s present; providing `epoch` yields better exposure math.
- For smoother curves, install **SciPy**. Without it, a moving average fallback is used (good enough for most cases).
- If the plot window doesn’t show, ensure your Matplotlib backend can create windows (the script sets `TkAgg`).

---

## Example Workflow

1. Train a model while logging rows to `loss_log.jsonl`.  
2. Run `python LossAnalyzer.py` and click **Load JSONL**.  
3. Inspect the plot, check **Slope**, **SNR**, **Steps since best**, **Exposure**.  
4. If **Plateau** with no new best for a while and exposure ≥ 18%, consider an early stop.

---

## Contributing

Issues and PRs are welcome. Ideas: CSV ingest, per‑epoch summaries, multi‑run comparison overlay, export PNG/SVG, headless batch reports.

---

## Acknowledgements

- Built with **Tkinter**, **Matplotlib**, and **NumPy** (optional **SciPy**, **pyperclip**).
- Loss smoothing and metrics design inspired by practical needs in small‑model LoRA/SFT/DPO training loops.
