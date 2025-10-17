
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LossAnalyzer.py — cleaned & optimized

Fixes / improvements:
- Define Analyze handler (no NameError) and wire buttons correctly.
- Remove emojis from titles/labels to avoid font warnings on Windows.
- Robust JSON/JSONL parsing (accepts single-quoted dicts, JSONL, pasted blobs).
- Actionable metrics: slope (last-N), MAD noise, SNR, best step, exposure %, LR info.
- Safer smoothing: uses SciPy's uniform_filter1d if available; otherwise a fast moving average.
- Clipboard export falls back gracefully if pyperclip isn't installed.
- Clearer GUI structure; only builds Tk when not invoked in CLI mode.
- CLI mode: `python LossAnalyzer.py path/to/log.jsonl` opens plot without GUI.
"""
import json
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Try SciPy smoothing; fallback to numpy moving average if SciPy isn't available
try:
    from scipy.ndimage import uniform_filter1d as _uniform_filter1d
    def smooth_series(y, k):
        k = max(1, int(k))
        return _uniform_filter1d(y, size=k, mode="nearest")
except Exception:
    def smooth_series(y, k):
        """Fast moving average fallback (same length as y)."""
        k = max(1, int(k))
        if k == 1 or len(y) < 2:
            return y.astype(float, copy=True)
        kernel = np.ones(k, dtype=float) / k
        pad = k // 2
        ypad = np.pad(y, (pad, k - 1 - pad), mode="edge")
        sm = np.convolve(ypad, kernel, mode="valid")
        return sm

# Optional clipboard support
try:
    import pyperclip  # type: ignore
    _HAS_PYPERCLIP = True
except Exception:
    _HAS_PYPERCLIP = False

# ----------------------------
# Globals filled by the GUI
# ----------------------------
output_labels = []
input_box: ScrolledText | None = None
output_frame: ttk.Frame | None = None

# ----------------------------
# Parsing
# ----------------------------
def parse_loss_input(raw_input: str):
    """
    Robustly parse logs from:
      - JSONL (one dict per line)
      - Console dumps with single quotes
      - Pasted blobs with {...}{...} etc.

    Returns a list[dict] with keys like 'loss', 'epoch', 'learning_rate', ...
    """
    parsed: list[dict] = []

    # First pass: JSONL one-per-line
    for line in raw_input.splitlines():
        line = line.strip()
        if not line:
            continue
        entry = None
        try:
            entry = json.loads(line)
        except Exception:
            # Accept single-quoted dicts
            try:
                fixed = re.sub(r"(?<!\\)'", '"', line)
                entry = json.loads(fixed)
            except Exception:
                entry = None
        if isinstance(entry, dict) and "loss" in entry:
            parsed.append(entry)

    if parsed:
        return parsed

    # Fallback: scan for {...} blocks inside the blob
    try:
        objs = re.findall(r"\{[^{}]+\}", raw_input)
        for obj in objs:
            fixed = obj.replace("'", '"')
            entry = json.loads(fixed)
            if isinstance(entry, dict) and "loss" in entry:
                parsed.append(entry)
    except Exception:
        pass

    return parsed

# ----------------------------
# Grading
# ----------------------------
def calculate_grade(metrics: dict) -> str:
    """
    Grade based on normalized slope (drop good), signal-to-noise ratio, and exposure.
    Expects: slope_norm (per 100 steps), snr, exposure_pct (0..1)
    """
    s = float(metrics.get("slope_norm", 0.0))
    snr = float(metrics.get("snr", 0.0))
    exposure = float(metrics.get("exposure_pct", 0.0))

    score = 0

    # Slope (per 100 steps): large negative is better
    if s <= -3.0:
        score += 4
    elif s <= -1.5:
        score += 3
    elif s <= -0.5:
        score += 2
    elif s <= -0.1:
        score += 1

    # SNR
    if snr >= 3.0:
        score += 3
    elif snr >= 2.0:
        score += 2
    elif snr >= 1.2:
        score += 1

    # Exposure floor bonuses (18% / 36%)
    if exposure >= 0.36:
        score += 2
    elif exposure >= 0.18:
        score += 1

    if score >= 8:
        return "A"
    if score >= 6:
        return "B"
    if score >= 4:
        return "C"
    return "D"

# ----------------------------
# Plotting & metrics
# ----------------------------
def plot_loss_curve(logs: list[dict]):
    global output_labels, output_frame

    # Extract
    steps = list(range(1, len(logs) + 1))
    losses = [e["loss"] for e in logs if "loss" in e]
    epochs = [e.get("epoch", 0.0) for e in logs if "loss" in e]
    lrs    = [e.get("learning_rate", None) for e in logs if "loss" in e]

    if not losses or len(losses) < 3:
        try:
            messagebox.showerror("Error", "Too few valid 'loss' values for analysis.")
        except Exception:
            print("Too few valid 'loss' values for analysis.")
        return

    # Exposure estimate (fraction of an epoch at the tail)
    last_epoch = float(epochs[-1]) if epochs else 0.0
    exposure = last_epoch - int(last_epoch) if last_epoch >= 1.0 else last_epoch
    exposure_pct = round(exposure * 100.0, 2)

    # Arrays
    x = np.array(steps, dtype=float)
    y = np.array(losses, dtype=float)

    # Smoothing
    k = 7 if len(y) >= 7 else max(3, (len(y) // 3) * 2 + 1)
    smoothed = smooth_series(y, k)

    # Recent window slope
    win = min(max(10, len(y) // 10), 200)
    xw = x[-win:]
    yw = smoothed[-win:]
    # Weighted linear fit (more weight on recent points)
    w = np.linspace(0.5, 1.0, num=len(xw))
    coeffs = np.polyfit(xw, yw, 1, w=w)
    slope = float(coeffs[0])  # change in loss per step (recent)
    slope_norm = slope * 100.0  # per 100 steps

    # Noise via robust MAD over window
    med = float(np.median(yw))
    mad = float(1.4826 * np.median(np.abs(yw - med)))
    noise = mad if mad > 1e-8 else float(np.std(yw))
    snr = float(abs(slope) / (noise + 1e-8))

    # Best loss & recency
    best_idx = int(np.argmin(y))
    best_loss = float(y[best_idx])
    steps_since_best = len(y) - (best_idx + 1)

    # Overall improvement
    drop_abs = float(y[0] - y[-1])
    drop_pct = 100.0 * drop_abs / max(y[0], 1e-8)

    # Predict reach of "ideal zone" (best + 0.05)
    eps = 0.05
    target = best_loss + eps
    if slope < -1e-6:
        steps_to_target = max(0, int((y[-1] - target) / -slope))
        predicted_stop = steps[-1] + steps_to_target
    else:
        predicted_stop = None

    # LR info
    lr_vals = [v for v in lrs if v is not None]
    lr_now = float(lr_vals[-1]) if lr_vals else 0.0
    lr_max = float(max(lr_vals)) if lr_vals else 0.0
    lr_mode = "dynamic" if len(set(lr_vals)) > 1 else "fixed"
    if lr_now >= 1e-3:
        lr_bucket = "High"
    elif lr_now >= 1e-4:
        lr_bucket = "Medium"
    else:
        lr_bucket = "Low"

    # Trend string (ASCII only to avoid font warnings)
    trend = (
        "Strong Drop" if slope_norm <= -3.0 else
        "Moderate Drop" if slope_norm <= -1.0 else
        "Weak Drop" if slope_norm < -0.1 else
        "Flat" if abs(slope_norm) <= 0.1 else
        "Weak Rise" if slope_norm < 1.0 else
        "Moderate Rise" if slope_norm < 3.0 else
        "Strong Rise"
    )

    # Grade
    grade = calculate_grade({
        "slope_norm": slope_norm,
        "snr": snr,
        "exposure_pct": exposure / 1.0,
    })

    # Early-stop suggestion (18% exposure floor)
    if exposure < 0.18:
        suggestion = f"Under exposure floor (seen {exposure_pct}%). Keep training to ≥ 18%."
    elif slope_norm <= -1.0 and snr >= 1.5:
        suggestion = "Healthy downward trend. Continue."
    elif abs(slope_norm) <= 0.2 and steps_since_best > max(25, win // 3):
        suggestion = f"Plateau: no new best in {steps_since_best} steps. Consider early stop soon."
    elif slope_norm > 0.5:
        suggestion = "Loss rising. Check LR, data, or stop."
    else:
        suggestion = "Stable. Monitor a bit longer."

    # ------------- UI metrics -------------
    labels = [
        ("Steps",             f"{len(steps):,}"),
        ("Current Loss",      f"{y[-1]:.4f}"),
        ("Best Loss",         f"{best_loss:.4f} @ {best_idx+1}"),
        ("Drop (total)",      f"-{drop_abs:.4f} ({drop_pct:.1f}%)"),
        ("Slope (last N)",    f"{slope_norm:+.3f} per 100 steps"),
        ("Noise (MAD)",       f"{noise:.4f}"),
        ("Signal/Noise",      f"{snr:.2f}"),
        ("Steps since best",  f"{steps_since_best}"),
        ("Exposure",          f"{exposure_pct:.2f}% (floor 18%)"),
        ("LR",                f"{lr_bucket} ({lr_mode})"),
        ("LR now / max",      f"{lr_now:.2e} / {lr_max:.2e}"),
        ("Trend",             trend),
        ("Grade",             grade),
        ("Suggestion",        suggestion),
    ]

    # wipe & render metrics (if GUI present)
    if output_frame is not None:
        for lbl in output_labels:
            try:
                lbl.destroy()
            except Exception:
                pass
        output_labels.clear()

        for i, (k, v) in enumerate(labels):
            lbl1 = ttk.Label(output_frame, text=k, anchor="w", font=("Segoe UI", 10))
            lbl1.grid(row=i, column=0, sticky="w", padx=(4, 2))
            lbl2 = ttk.Label(output_frame, text=v, anchor="w", font=("Segoe UI", 10))
            lbl2.grid(row=i, column=1, sticky="w", padx=(2, 10))
            output_labels.extend([lbl1, lbl2])

    # ------------- Plot -------------
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(steps, y, label="Loss", alpha=0.25)
    ax1.plot(steps, smoothed, label=f"Smoothed (k={k})")

    # Best-loss marker
    ax1.scatter([best_idx + 1], [best_loss], color="green", s=36, zorder=3, label="Best")

    # Ideal zone band (best .. best+0.05)
    ideal_min = best_loss
    ax1.axhspan(ideal_min, ideal_min + 0.05, color="green", alpha=0.08, label="Ideal Zone (+0.05)")

    # Outlier highlights vs smooth (last window)
    if len(yw) > 1:
        std_dev = float(np.std(yw))
        if std_dev > 0:
            for i in range(win):
                idx = len(steps) - win + i
                if 0 < idx < len(smoothed):
                    if abs(y[idx] - smoothed[idx]) > std_dev * 1.5:
                        ax1.axvspan(idx - 0.5, idx + 0.5, color="red", alpha=0.05)

    # Epoch boundary markers (approx; based on epoch change)
    for i in range(1, len(epochs)):
        if int(epochs[i - 1] * 100) != int(epochs[i] * 100):
            ax1.axvline(x=i + 1, color="gray", linestyle="--", alpha=0.25)

    ax1.set_title("Loss (best, trend, ideal zone)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # LR overlay (if available)
    if lr_vals:
        ax2 = ax1.twinx()
        ax2.plot(steps, lr_vals, alpha=0.25, linestyle="--")
        ax2.set_ylabel("Learning Rate")

    # Predicted stop
    if predicted_stop is not None:
        ax1.axvline(predicted_stop, color="purple", linestyle=":", alpha=0.6)
        ax1.text(predicted_stop, ax1.get_ylim()[0], "pred stop", rotation=90,
                 va="bottom", ha="right", fontsize=8, color="purple")

    plt.tight_layout()
    plt.show()

# ----------------------------
# GUI helpers
# ----------------------------
def load_jsonl_file():
    """Prompt for a .jsonl file and load its contents into the input box."""
    global input_box
    file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
    if not file_path or input_box is None:
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        input_box.delete("1.0", "end")
        input_box.insert("1.0", raw)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def copy_metrics_to_clipboard():
    """Copy current metrics table into clipboard as 'Metric | Value' lines."""
    global output_frame
    if output_frame is None:
        return
    rows = []
    # children are rendered as alternating labels (metric, value)
    children = [w for w in output_frame.winfo_children() if isinstance(w, ttk.Label)]
    for i in range(0, len(children), 2):
        k = children[i].cget("text")
        v = children[i + 1].cget("text") if i + 1 < len(children) else ""
        rows.append(f"{k}: {v}")
    text = "\n".join(rows)
    if _HAS_PYPERCLIP:
        try:
            pyperclip.copy(text)
            messagebox.showinfo("Copied", "Metrics copied to clipboard.")
        except Exception as e:
            messagebox.showwarning("Clipboard", f"Could not copy: {e}")
    else:
        # Fallback: put it back into the input box so user can copy
        if input_box is not None:
            input_box.insert("end", "\n\n# Metrics\n" + text + "\n")
            input_box.see("end")
        messagebox.showinfo("Clipboard", "pyperclip not installed; metrics appended to the input box.")

def export_to_markdown():
    """Export the metrics table as a Markdown file."""
    global output_frame
    if output_frame is None:
        return
    children = [w for w in output_frame.winfo_children() if isinstance(w, ttk.Label)]
    lines = ["| Metric | Value |", "|--------|--------|"]
    for i in range(0, len(children), 2):
        k = children[i].cget("text")
        v = children[i + 1].cget("text") if i + 1 < len(children) else ""
        lines.append(f"| {k} | {v} |")
    save_path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")])
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Saved", f"Saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ----------------------------
# GUI setup
# ----------------------------
def launch_gui():
    global input_box, output_frame

    root = tk.Tk()
    root.title("Loss Curve Analyzer")
    root.geometry("900x640")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    mainframe = ttk.Frame(root, padding=10)
    mainframe.pack(fill="both", expand=True)

    # Input box
    input_box = ScrolledText(mainframe, width=120, height=16, font=("Consolas", 11), wrap="none")
    input_box.grid(row=0, column=0, pady=(6, 10), sticky="nsew")

    # Output metrics
    output_frame = ttk.Frame(mainframe)
    output_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

    # Buttons
    button_frame = ttk.Frame(mainframe)
    button_frame.grid(row=2, column=0, sticky="ew")

    # --- Handlers ---
    def handle_analyze(event=None):
        raw = input_box.get("1.0", "end").strip()
        logs = parse_loss_input(raw)
        if not logs:
            messagebox.showwarning("No logs", "Paste some JSON/JSONL training rows first.")
            return
        plot_loss_curve(logs)

    ttk.Button(button_frame, text="Load JSONL", command=load_jsonl_file).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Copy Metrics", command=copy_metrics_to_clipboard).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Export Markdown", command=export_to_markdown).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Analyze Now", command=handle_analyze).pack(side="left", padx=4)

    # Keyboard shortcut: Ctrl+Enter to analyze
    root.bind("<Control-Return>", handle_analyze)

    # Resizing behavior
    mainframe.rowconfigure(0, weight=1)
    mainframe.rowconfigure(1, weight=0)
    mainframe.columnconfigure(0, weight=1)

    root.mainloop()

# ----------------------------
# Entrypoint
# ----------------------------
def _cli_main():
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".jsonl"):
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(2)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        logs = parse_loss_input(raw)
        if logs:
            plot_loss_curve(logs)
        else:
            print("No valid log rows found.")
    else:
        launch_gui()

if __name__ == "__main__":
    _cli_main()
