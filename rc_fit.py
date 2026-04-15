"""
RC Discharge Curve Fitter
Fits V(t) = A * exp(-w * t) + C to time-voltage CSV data,
then computes capacitance from C_cap = 1 / (w * R).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.optimize import curve_fit
import csv
import os


# ── model ────────────────────────────────────────────────────────────────
def exp_decay(t, A, w, C):
    return A * np.exp(-w * t) + C


# ── CSV loader ───────────────────────────────────────────────────────────
TIME_COLS = {
    "time_s":   1.0,
    "time_ms":  1e-3,
    "time_mus": 1e-6,
    "time_ns":  1e-9,
}

VOLT_COLS = {
    "voltage":    1.0,
    "voltage_v":  1.0,
    "voltage_mv": 1e-3,
}


def load_csv(path):
    """Return (time_s, voltage_V) numpy arrays from a CSV file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]

        # find time column
        time_idx, time_scale = None, None
        for name, scale in TIME_COLS.items():
            if name in header:
                time_idx = header.index(name)
                time_scale = scale
                break

        # find voltage column
        volt_idx, volt_scale = None, None
        for name, scale in VOLT_COLS.items():
            if name in header:
                volt_idx = header.index(name)
                volt_scale = scale
                break

        if time_idx is None or volt_idx is None:
            raise ValueError(
                "CSV must contain a time column (time_s, time_ms, time_mus, "
                "time_ns) and a voltage column (voltage, voltage_V, voltage_mV)."
            )

        times, volts = [], []
        for row in reader:
            if not row:
                continue
            times.append(float(row[time_idx]) * time_scale)
            volts.append(float(row[volt_idx]) * volt_scale)

    return np.array(times), np.array(volts)


# ── GUI ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RC Exponential Fitter")
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

        # --- fit button ---
        ttk.Button(frame, text="Fit", command=self._fit).grid(
            row=2, column=0, columnspan=3, pady=(12, 0)
        )

        # --- results ---
        self.result_text = tk.Text(
            frame, height=14, width=60, state="disabled", font=("Consolas", 10)
        )
        self.result_text.grid(row=3, column=0, columnspan=3, pady=(8, 0))

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
        # validate inputs
        csv_path = self.file_var.get().strip()
        if not csv_path or not os.path.isfile(csv_path):
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return

        try:
            R = float(self.r_var.get())
            if R <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Enter a positive number for resistance.")
            return

        # load data
        try:
            t, V = load_csv(csv_path)
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))
            return

        if len(t) < 4:
            messagebox.showerror("Error", "Need at least 4 data points for the fit.")
            return

        # initial guesses: A ≈ V_range, w > 0, C ≈ final voltage
        A0 = V[0] - V[-1]
        C0 = V[-1]
        t_span = t[-1] - t[0]
        w0 = 1.0 / t_span if t_span > 0 else 1.0

        try:
            popt, pcov = curve_fit(
                exp_decay, t, V,
                p0=[A0, w0, C0],
                maxfev=50000,
            )
        except RuntimeError as e:
            messagebox.showerror("Fit Error", f"Curve fit failed:\n{e}")
            return

        A_fit, w_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))          # 1-sigma uncertainties
        A_err, w_err, C_err = perr

        # capacitance: C_cap = 1 / (w * R)
        C_cap = 1.0 / (w_fit * R)
        # propagate uncertainty in w  (R assumed exact)
        C_cap_err = w_err / (w_fit**2 * R)

        # residual quality
        residuals = V - exp_decay(t, *popt)
        chi2 = np.sum(residuals**2)
        rmse = np.sqrt(chi2 / len(t))

        lines = [
            "═══════════════  FIT RESULTS  ═══════════════",
            "",
            f"  V(t) = A·exp(−w·t) + C",
            "",
            f"  A   = {A_fit: .6g}  ±  {A_err:.3g}  V",
            f"  w   = {w_fit: .6g}  ±  {w_err:.3g}  1/s",
            f"  C   = {C_fit: .6g}  ±  {C_err:.3g}  V",
            "",
            f"  R (given)       = {R:.6g}  Ω",
            f"  τ = 1/w         = {1/w_fit:.6g}  ±  {w_err/w_fit**2:.3g}  s",
            f"  Capacitance     = {C_cap:.6g}  ±  {C_cap_err:.3g}  F",
            "",
            _auto_prefix("Capacitance", C_cap, C_cap_err),
            "",
            f"  RMSE            = {rmse:.4g}  V",
            f"  Data points     = {len(t)}",
            "",
            "═════════════════════════════════════════════",
        ]
        self._set_result("\n".join(lines))


def _auto_prefix(label, val, err):
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
            return f"  {label:14s}  = {val/scale:.4g}  ±  {err/scale:.3g}  {sym}F"
    return f"  {label:14s}  = {val:.4g}  ±  {err:.3g}  F"


if __name__ == "__main__":
    App().mainloop()
