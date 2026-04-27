#!/usr/bin/env python
"""
Measure capacitance from the frequency response of an RC low-pass filter.

The user supplies a CSV of (frequency, Vpp_capacitor) data and the known
resistance R.  The script fits the model

    Vc_pp(f) = Vin_pp / sqrt(1 + (2*pi*f*R*C)^2)

to extract C (and Vin_pp as a nuisance parameter).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.optimize import curve_fit
import csv
import os


# ── model ────────────────────────────────────────────────────────────────
def vc_model(f, Vin, fc):
    """Voltage across the capacitor in a series RC low-pass filter.

    Parameterised by corner frequency fc = 1/(2πRC) so that both fit
    parameters (Vin ~ volts, fc ~ Hz) are on a similar numeric scale.
    """
    return Vin / np.sqrt(1.0 + (f / fc) ** 2)


# ── CSV loader ───────────────────────────────────────────────────────────
FREQ_COLS = {
    "frequency_hz":  1.0,
    "freq_hz":       1.0,
    "frequency":     1.0,
    "freq":          1.0,
    "frequency_khz": 1e3,
    "freq_khz":      1e3,
    "frequency_mhz": 1e6,
    "freq_mhz":      1e6,
}

VOLT_COLS = {
    "vpp":        1.0,
    "vpp_v":      1.0,
    "voltage":    1.0,
    "voltage_v":  1.0,
    "vpp_mv":     1e-3,
    "voltage_mv": 1e-3,
}


def load_csv(path):
    """Return (frequency_Hz, Vpp_V) numpy arrays from a CSV file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]

        # find frequency column
        freq_idx, freq_scale = None, None
        for name, scale in FREQ_COLS.items():
            if name in header:
                freq_idx = header.index(name)
                freq_scale = scale
                break

        # find voltage column
        volt_idx, volt_scale = None, None
        for name, scale in VOLT_COLS.items():
            if name in header:
                volt_idx = header.index(name)
                volt_scale = scale
                break

        if freq_idx is None or volt_idx is None:
            raise ValueError(
                "CSV must contain a frequency column "
                "(frequency_Hz, freq_Hz, frequency_kHz, …) and a voltage "
                "column (Vpp, Vpp_V, voltage, Vpp_mV, …)."
            )

        freqs, volts = [], []
        for row in reader:
            if not row:
                continue
            freqs.append(float(row[freq_idx]) * freq_scale)
            volts.append(float(row[volt_idx]) * volt_scale)

    return np.array(freqs), np.array(volts)


# ── GUI ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RC Low-Pass Filter — Capacitance Fitter")
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=16)
        frame.grid()

        # --- file picker ---
        ttk.Label(frame, text="CSV File:").grid(row=0, column=0, sticky="e")
        self.file_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.file_var, width=48).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(frame, text="Browse…", command=self._browse).grid(
            row=0, column=2
        )

        # --- resistance input ---
        ttk.Label(frame, text="Resistance (Ω):").grid(
            row=1, column=0, sticky="e", pady=(8, 0)
        )
        self.r_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.r_var, width=20).grid(
            row=1, column=1, sticky="w", padx=4, pady=(8, 0)
        )

        # --- optional Vin input ---
        ttk.Label(frame, text="Vin pp (V) [optional]:").grid(
            row=2, column=0, sticky="e", pady=(8, 0)
        )
        self.vin_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.vin_var, width=20).grid(
            row=2, column=1, sticky="w", padx=4, pady=(8, 0)
        )

        # --- fit button ---
        ttk.Button(frame, text="Fit", command=self._fit).grid(
            row=3, column=0, columnspan=3, pady=(12, 0)
        )

        # --- results ---
        self.result_text = tk.Text(
            frame, height=18, width=64, state="disabled", font=("Consolas", 10)
        )
        self.result_text.grid(row=4, column=0, columnspan=3, pady=(8, 0))

    # ── helpers ──────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def _fit(self):
        # validate CSV path
        csv_path = self.file_var.get().strip()
        if not csv_path or not os.path.isfile(csv_path):
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return

        # validate resistance
        try:
            R = float(self.r_var.get())
            if R <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Enter a positive number for resistance.")
            return

        # optional fixed Vin
        vin_fixed = None
        vin_str = self.vin_var.get().strip()
        if vin_str:
            try:
                vin_fixed = float(vin_str)
                if vin_fixed <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Vin pp must be a positive number (or leave blank).")
                return

        # load data
        try:
            freq, Vpp = load_csv(csv_path)
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))
            return

        if len(freq) < 3:
            messagebox.showerror("Error", "Need at least 3 data points for the fit.")
            return

        # sort by frequency
        order = np.argsort(freq)
        freq, Vpp = freq[order], Vpp[order]

        # ── estimate initial fc from data ─────────────────────────
        # find the frequency closest to the -3 dB point (Vpp ≈ 0.707 * Vmax)
        Vmax = np.max(Vpp)
        target = 0.707 * Vmax
        fc0 = float(np.interp(target, Vpp[::-1], freq[::-1]))  # Vpp decreasing
        if fc0 <= 0:
            fc0 = float(np.median(freq))

        # ── curve fit ────────────────────────────────────────────────
        if vin_fixed is not None:
            # fit only fc, with Vin fixed
            def model_1p(f, fc):
                return vc_model(f, vin_fixed, fc)

            try:
                popt, pcov = curve_fit(
                    model_1p, freq, Vpp,
                    p0=[fc0],
                    bounds=(0, np.inf),
                    maxfev=50000,
                )
            except RuntimeError as e:
                messagebox.showerror("Fit Error", f"Curve fit failed:\n{e}")
                return

            fc_fit = popt[0]
            fc_err = np.sqrt(pcov[0, 0])
            Vin_fit = vin_fixed
            Vin_err = 0.0
        else:
            # fit both Vin and fc
            def model_2p(f, Vin, fc):
                return vc_model(f, Vin, fc)

            Vin0 = float(Vmax)
            try:
                popt, pcov = curve_fit(
                    model_2p, freq, Vpp,
                    p0=[Vin0, fc0],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=50000,
                )
            except RuntimeError as e:
                messagebox.showerror("Fit Error", f"Curve fit failed:\n{e}")
                return

            Vin_fit, fc_fit = popt
            perr = np.sqrt(np.diag(pcov))
            Vin_err, fc_err = perr

        # derived quantities
        C_fit = 1.0 / (2.0 * np.pi * fc_fit * R)
        # propagate fc uncertainty → C uncertainty:  C = 1/(2π fc R)
        # dC/dfc = -1/(2π fc² R)  →  σ_C = σ_fc / (2π fc² R)
        C_err = fc_err / (2.0 * np.pi * fc_fit**2 * R)

        # residual quality
        Vpp_pred = vc_model(freq, Vin_fit, fc_fit)
        residuals = Vpp - Vpp_pred
        chi2 = np.sum(residuals**2)
        rmse = np.sqrt(chi2 / len(freq))

        lines = [
            "══════════════  LOW-PASS FIT RESULTS  ═══════════════",
            "",
            "  Model:  Vc_pp(f) = Vin / √(1 + (2πfRC)²)",
            "",
            f"  Vin_pp  = {Vin_fit: .6g}  ±  {Vin_err:.3g}  V"
            + ("  (fixed)" if vin_fixed else ""),
            f"  R (given)      = {R:.6g}  Ω",
            "",
            _auto_prefix("Capacitance", C_fit, C_err),
            _auto_prefix("f_corner   ", fc_fit, fc_err, unit="Hz"),
            "",
            f"  RMSE           = {rmse:.4g}  V",
            f"  Data points    = {len(freq)}",
            "",
            "  ── Data vs Fit ──",
            f"  {'f (Hz)':>12s}  {'Vpp meas':>10s}  {'Vpp fit':>10s}  {'resid':>10s}",
        ]
        for fi, vm, vf, r in zip(freq, Vpp, Vpp_pred, residuals):
            lines.append(
                f"  {fi:12.4g}  {vm:10.4g}  {vf:10.4g}  {r:+10.4g}"
            )
        lines.append("")
        lines.append("═════════════════════════════════════════════════════")
        self._set_result("\n".join(lines))


def _auto_prefix(label, val, err, unit="F"):
    """Express value ± error with a sensible SI prefix."""
    prefixes = [
        (1e-15, "f"),
        (1e-12, "p"),
        (1e-9,  "n"),
        (1e-6,  "µ"),
        (1e-3,  "m"),
        (1.0,   ""),
        (1e3,   "k"),
        (1e6,   "M"),
    ]
    abs_val = abs(val)
    for scale, sym in prefixes:
        if abs_val < scale * 1000:
            return f"  {label:14s}  = {val/scale:.4g}  ±  {err/scale:.3g}  {sym}{unit}"
    return f"  {label:14s}  = {val:.4g}  ±  {err:.3g}  {unit}"


if __name__ == "__main__":
    App().mainloop()
