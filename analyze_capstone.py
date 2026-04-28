from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
SCOPE_RESISTANCE_OHM = 1_000_000.0
VOLTAGE_SIGMA_FLOOR_V = 0.020
EPSILON_0_F_PER_M = 8.8541878128e-12


@dataclass(frozen=True)
class RunConfig:
    key: str
    label: str
    path: str
    resistor_ohm: float
    material: str
    kind: str


@dataclass(frozen=True)
class FrequencyRunConfig:
    key: str
    label: str
    path: str
    resistor_ohm: float
    kind: str


RUNS = [
    RunConfig("air_no", "Air: No capacitor", "AirnoCapacitor.csv", 100_000.0, "Air", "No"),
    RunConfig("air_with_1", "Air: With capacitor 1", "AirwithCapacitor.csv", 100_000.0, "Air", "With"),
    RunConfig("air_with_2", "Air: With capacitor 2", "AirWithCapacitance2.csv", 100_000.0, "Air", "With"),
    RunConfig("glass_no", "Glass: No capacitor", "GlassNo.csv", 9_810.0, "Glass", "No"),
    RunConfig("glass_with", "Glass: With capacitor", "GlassWith.csv", 9_810.0, "Glass", "With"),
    RunConfig("acrylic_no", "Acrylic: No capacitor", "AcrylicNoCapactiror.csv", 9_810.0, "Acrylic", "No"),
    RunConfig("acrylic_with", "Acrylic: With capacitor", "AcyrlicWithCapacitor.csv", 9_810.0, "Acrylic", "With"),
]

FREQUENCY_RUNS = [
    FrequencyRunConfig(
        "air_freq_no",
        "Air frequency sweep: No capacitor",
        "HighPassNoCapacitor.csv",
        100_000.0,
        "No",
    ),
    FrequencyRunConfig(
        "air_freq_with",
        "Air frequency sweep: With capacitor",
        "HighPassWithCapacitor.csv",
        100_000.0,
        "With",
    ),
]


def effective_resistance(resistor_ohm: float) -> float:
    return 1.0 / (1.0 / resistor_ohm + 1.0 / SCOPE_RESISTANCE_OHM)


def decay_model(time_s: np.ndarray, amplitude_v: float, tau_s: float, offset_v: float) -> np.ndarray:
    return amplitude_v * np.exp(-time_s / tau_s) + offset_v


def low_pass_model(frequency_hz: np.ndarray, low_frequency_vpp: float, corner_hz: float) -> np.ndarray:
    return low_frequency_vpp / np.sqrt(1.0 + (frequency_hz / corner_hz) ** 2)


def load_transient_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = [item.strip().lower() for item in next(reader)]
        time_scale = 1e-6 if "mus" in header[0] else 1.0
        voltage_scale = 1e-3 if "mv" in header[1] else 1.0

        rows: list[tuple[float, float]] = []
        for row in reader:
            if not row:
                continue
            rows.append((float(row[0]) * time_scale, float(row[1]) * voltage_scale))

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1]


def load_frequency_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = [item.strip().lower() for item in next(reader)]
        frequency_index = header.index("frequency_hz")
        voltage_index = header.index("vpp_v")

        rows: list[tuple[float, float]] = []
        for row in reader:
            if not row:
                continue
            rows.append((float(row[frequency_index]), float(row[voltage_index])))

    data = np.array(rows, dtype=float)
    order = np.argsort(data[:, 0])
    data = data[order]
    return data[:, 0], data[:, 1]


def initial_tau_guess(time_s: np.ndarray, voltage_v: np.ndarray) -> float:
    amplitude = voltage_v[0] - voltage_v[-1]
    offset = voltage_v[-1]
    target = offset + amplitude / math.e
    index = int(np.argmin(np.abs(voltage_v - target)))
    return max(float(time_s[index]), float((time_s[-1] - time_s[0]) / 3.0), 1e-9)


def fit_run(config: RunConfig) -> dict[str, float | np.ndarray | str]:
    time_s, voltage_v = load_transient_csv(ROOT / config.path)
    p0 = [float(voltage_v[0] - voltage_v[-1]), initial_tau_guess(time_s, voltage_v), float(voltage_v[-1])]

    bounds = ([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf])
    first_params, _ = curve_fit(
        decay_model,
        time_s,
        voltage_v,
        p0=p0,
        bounds=bounds,
        maxfev=100_000,
    )
    first_residuals = voltage_v - decay_model(time_s, *first_params)
    dof = len(voltage_v) - len(first_params)
    residual_sigma = math.sqrt(float(np.sum(first_residuals**2)) / max(dof, 1))
    sigma_v = max(VOLTAGE_SIGMA_FLOOR_V, residual_sigma)

    params, covariance = curve_fit(
        decay_model,
        time_s,
        voltage_v,
        p0=first_params,
        sigma=np.full_like(voltage_v, sigma_v),
        absolute_sigma=True,
        bounds=bounds,
        maxfev=100_000,
    )
    errors = np.sqrt(np.diag(covariance))
    model_v = decay_model(time_s, *params)
    residuals = voltage_v - model_v
    chi2 = float(np.sum((residuals / sigma_v) ** 2))
    reduced_chi2 = chi2 / dof
    r_eff = effective_resistance(config.resistor_ohm)
    capacitance_f = float(params[1] / r_eff)
    capacitance_err_f = float(errors[1] / r_eff)

    return {
        "key": config.key,
        "label": config.label,
        "path": config.path,
        "material": config.material,
        "kind": config.kind,
        "resistor_ohm": config.resistor_ohm,
        "r_eff_ohm": r_eff,
        "time_s": time_s,
        "voltage_v": voltage_v,
        "amplitude_v": float(params[0]),
        "amplitude_err_v": float(errors[0]),
        "tau_s": float(params[1]),
        "tau_err_s": float(errors[1]),
        "offset_v": float(params[2]),
        "offset_err_v": float(errors[2]),
        "sigma_v": sigma_v,
        "residuals_v": residuals,
        "chi2": chi2,
        "dof": float(dof),
        "reduced_chi2": reduced_chi2,
        "capacitance_f": capacitance_f,
        "capacitance_err_f": capacitance_err_f,
    }


def initial_corner_guess(frequency_hz: np.ndarray, voltage_vpp: np.ndarray) -> float:
    maximum = float(np.max(voltage_vpp))
    target = maximum / math.sqrt(2.0)
    descending_voltage = voltage_vpp[::-1]
    descending_frequency = frequency_hz[::-1]
    if float(np.min(voltage_vpp)) <= target <= maximum:
        return float(np.interp(target, descending_voltage, descending_frequency))
    return float(np.median(frequency_hz))


def fit_frequency_run(config: FrequencyRunConfig) -> dict[str, float | np.ndarray | str]:
    frequency_hz, voltage_vpp = load_frequency_csv(ROOT / config.path)
    p0 = [float(np.max(voltage_vpp)), initial_corner_guess(frequency_hz, voltage_vpp)]

    bounds = ([0.0, 0.0], [np.inf, np.inf])
    first_params, _ = curve_fit(
        low_pass_model,
        frequency_hz,
        voltage_vpp,
        p0=p0,
        bounds=bounds,
        maxfev=100_000,
    )
    first_residuals = voltage_vpp - low_pass_model(frequency_hz, *first_params)
    dof = len(voltage_vpp) - len(first_params)
    residual_sigma = math.sqrt(float(np.sum(first_residuals**2)) / max(dof, 1))
    sigma_v = max(VOLTAGE_SIGMA_FLOOR_V, residual_sigma)

    params, covariance = curve_fit(
        low_pass_model,
        frequency_hz,
        voltage_vpp,
        p0=first_params,
        sigma=np.full_like(voltage_vpp, sigma_v),
        absolute_sigma=True,
        bounds=bounds,
        maxfev=100_000,
    )
    errors = np.sqrt(np.diag(covariance))
    model_vpp = low_pass_model(frequency_hz, *params)
    residuals = voltage_vpp - model_vpp
    chi2 = float(np.sum((residuals / sigma_v) ** 2))
    reduced_chi2 = chi2 / dof
    r_eff = effective_resistance(config.resistor_ohm)
    capacitance_f = float(1.0 / (2.0 * math.pi * params[1] * r_eff))
    capacitance_err_f = float(errors[1] / (2.0 * math.pi * params[1] ** 2 * r_eff))

    return {
        "key": config.key,
        "label": config.label,
        "path": config.path,
        "kind": config.kind,
        "resistor_ohm": config.resistor_ohm,
        "r_eff_ohm": r_eff,
        "frequency_hz": frequency_hz,
        "voltage_vpp": voltage_vpp,
        "low_frequency_vpp": float(params[0]),
        "low_frequency_err_vpp": float(errors[0]),
        "corner_hz": float(params[1]),
        "corner_err_hz": float(errors[1]),
        "sigma_v": sigma_v,
        "residuals_v": residuals,
        "chi2": chi2,
        "dof": float(dof),
        "reduced_chi2": reduced_chi2,
        "capacitance_f": capacitance_f,
        "capacitance_err_f": capacitance_err_f,
    }


def format_pm(value: float, err: float, scale: float = 1.0, digits: int = 2) -> str:
    return f"{value * scale:.{digits}f} +/- {err * scale:.{digits}f}"


def plot_material(material: str, fits: list[dict[str, float | np.ndarray | str]]) -> None:
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for color, fit in zip(colors, fits):
        time_us = np.asarray(fit["time_s"]) * 1e6
        voltage_v = np.asarray(fit["voltage_v"])
        sigma_v = float(fit["sigma_v"])
        label = str(fit["label"]).replace(f"{material}: ", "")
        ax.errorbar(
            time_us,
            voltage_v,
            yerr=sigma_v,
            fmt="o",
            color=color,
            alpha=0.82,
            capsize=3,
            markersize=4.5,
            label=f"{label} data",
        )
        smooth_time_s = np.linspace(float(np.min(fit["time_s"])), float(np.max(fit["time_s"])), 400)
        ax.plot(
            smooth_time_s * 1e6,
            decay_model(
                smooth_time_s,
                float(fit["amplitude_v"]),
                float(fit["tau_s"]),
                float(fit["offset_v"]),
            ),
            color=color,
            linewidth=2.0,
            label=rf"{label} fit, $\tau={float(fit['tau_s']) * 1e6:.2f}\,\mu$s",
        )

    ax.set_title(f"{material}: Transient Fits")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(IMAGES / f"{material.lower()}_transient_fit.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for color, fit in zip(colors, fits):
        time_us = np.asarray(fit["time_s"]) * 1e6
        residuals_v = np.asarray(fit["residuals_v"])
        sigma_v = float(fit["sigma_v"])
        label = str(fit["label"]).replace(f"{material}: ", "")
        ax.errorbar(
            time_us,
            residuals_v,
            yerr=sigma_v,
            fmt="o",
            color=color,
            alpha=0.82,
            capsize=3,
            markersize=4.5,
            label=f"{label} residuals",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"{material}: Residuals")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Residual (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(IMAGES / f"{material.lower()}_transient_residuals.png", dpi=300)
    plt.close(fig)


def plot_frequency_response(fits: list[dict[str, float | np.ndarray | str]]) -> None:
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for color, fit in zip(colors, fits):
        frequency_hz = np.asarray(fit["frequency_hz"])
        voltage_vpp = np.asarray(fit["voltage_vpp"])
        sigma_v = float(fit["sigma_v"])
        label = str(fit["label"]).replace("Air frequency sweep: ", "")
        ax.errorbar(
            frequency_hz,
            voltage_vpp,
            yerr=sigma_v,
            fmt="o",
            color=color,
            alpha=0.82,
            capsize=3,
            markersize=4.5,
            label=f"{label} data",
        )
        smooth_frequency_hz = np.logspace(
            math.log10(float(np.min(frequency_hz))),
            math.log10(float(np.max(frequency_hz))),
            500,
        )
        ax.plot(
            smooth_frequency_hz,
            low_pass_model(
                smooth_frequency_hz,
                float(fit["low_frequency_vpp"]),
                float(fit["corner_hz"]),
            ),
            color=color,
            linewidth=2.0,
            label=rf"{label} fit, $f_c={float(fit['corner_hz']) / 1000:.2f}\,$kHz",
        )

    ax.set_xscale("log")
    ax.set_title("Air: Frequency-Response Fits")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$V_{\mathrm{pp}}$ (V)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(IMAGES / "air_frequency_response_fit.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for color, fit in zip(colors, fits):
        frequency_hz = np.asarray(fit["frequency_hz"])
        residuals_v = np.asarray(fit["residuals_v"])
        sigma_v = float(fit["sigma_v"])
        label = str(fit["label"]).replace("Air frequency sweep: ", "")
        ax.errorbar(
            frequency_hz,
            residuals_v,
            yerr=sigma_v,
            fmt="o",
            color=color,
            alpha=0.82,
            capsize=3,
            markersize=4.5,
            label=f"{label} residuals",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_title("Air: Frequency-Response Residuals")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual (V)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(IMAGES / "air_frequency_response_residuals.png", dpi=300)
    plt.close(fig)


def plot_summary(finals: dict[str, tuple[float, float]]) -> None:
    labels = list(finals)
    values_pf = np.array([finals[label][0] for label in labels]) * 1e12
    errors_pf = np.array([finals[label][1] for label in labels]) * 1e12

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    colors = ["tab:blue", "tab:green", "tab:orange"]
    positions = np.arange(len(labels))
    ax.bar(positions, values_pf, yerr=errors_pf, capsize=5, color=colors[: len(labels)], alpha=0.82)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Corrected sample capacitance (pF)")
    ax.set_title("Final Capacitance After Scope-Capacitance Subtraction")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(IMAGES / "final_capacitance_summary.png", dpi=300)
    plt.close(fig)


def plot_permittivity_summary(permittivities: dict[str, tuple[float, float]]) -> None:
    labels = list(permittivities)
    values = np.array([permittivities[label][0] for label in labels])
    errors = np.array([permittivities[label][1] for label in labels])

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    colors = ["tab:blue", "tab:green", "tab:orange"]
    positions = np.arange(len(labels))
    ax.bar(positions, values, yerr=errors, capsize=5, color=colors[: len(labels)], alpha=0.82)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label=r"$\kappa=1$")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Relative permittivity, $\kappa$")
    ax.set_title("Dielectric Constants From Corrected Capacitance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(IMAGES / "permittivity_summary.png", dpi=300)
    plt.close(fig)


def corrected_capacitance(
    baseline: dict[str, float | np.ndarray | str],
    with_run: dict[str, float | np.ndarray | str],
) -> tuple[float, float]:
    value = float(with_run["capacitance_f"]) - float(baseline["capacitance_f"])
    error = math.sqrt(float(with_run["capacitance_err_f"]) ** 2 + float(baseline["capacitance_err_f"]) ** 2)
    return value, error


def weighted_mean(values: list[float], errors: list[float]) -> tuple[float, float, float]:
    weights = 1.0 / np.square(errors)
    mean = float(np.sum(weights * np.array(values)) / np.sum(weights))
    formal_error = math.sqrt(float(1.0 / np.sum(weights)))
    if len(values) > 1:
        chi2 = float(np.sum(((np.array(values) - mean) / np.array(errors)) ** 2))
        reduced_chi2 = chi2 / (len(values) - 1)
        scaled_error = formal_error * math.sqrt(max(1.0, reduced_chi2))
    else:
        reduced_chi2 = 0.0
        scaled_error = formal_error
    return mean, scaled_error, reduced_chi2


def dielectric_permittivity(
    capacitance_f: float,
    capacitance_err_f: float,
    area_m2: float,
    thickness_m: float,
    area_err_m2: float = 0.0,
    thickness_err_m: float = 0.0,
) -> tuple[float, float, float, float]:
    epsilon = capacitance_f * thickness_m / area_m2
    rel_terms = [
        capacitance_err_f / capacitance_f if capacitance_f else 0.0,
        area_err_m2 / area_m2 if area_m2 else 0.0,
        thickness_err_m / thickness_m if thickness_m else 0.0,
    ]
    epsilon_err = abs(epsilon) * math.sqrt(sum(term**2 for term in rel_terms))
    kappa = epsilon / EPSILON_0_F_PER_M
    kappa_err = epsilon_err / EPSILON_0_F_PER_M
    return epsilon, epsilon_err, kappa, kappa_err


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    fits = {config.key: fit_run(config) for config in RUNS}
    frequency_fits = {config.key: fit_frequency_run(config) for config in FREQUENCY_RUNS}

    for material in ["Air", "Glass", "Acrylic"]:
        material_fits = [fit for fit in fits.values() if fit["material"] == material]
        plot_material(material, material_fits)
    plot_frequency_response([frequency_fits[config.key] for config in FREQUENCY_RUNS])

    air_1 = corrected_capacitance(fits["air_no"], fits["air_with_1"])
    air_2 = corrected_capacitance(fits["air_no"], fits["air_with_2"])
    air_mean, air_mean_err, air_weighted_red_chi2 = weighted_mean(
        [air_1[0], air_2[0]],
        [air_1[1], air_2[1]],
    )
    glass = corrected_capacitance(fits["glass_no"], fits["glass_with"])
    acrylic = corrected_capacitance(fits["acrylic_no"], fits["acrylic_with"])
    air_frequency = corrected_capacitance(
        frequency_fits["air_freq_no"],
        frequency_fits["air_freq_with"],
    )

    finals = {
        "Air": (air_mean, air_mean_err),
        "Glass": glass,
        "Acrylic": acrylic,
    }
    plot_summary(finals)

    CALIPER_M = 2e-5  # 0.02 mm caliper uncertainty

    air_radius_m = 0.10
    air_area_m2 = math.pi * air_radius_m**2
    air_area_err_m2 = 2.0 * math.pi * air_radius_m * CALIPER_M  # dA/dr = 2πr
    air_thickness_m = 0.0025
    air_thickness_err_m = CALIPER_M

    glass_area_m2 = 0.34 * 0.48
    glass_area_err_m2 = glass_area_m2 * math.sqrt((CALIPER_M / 0.34) ** 2 + (CALIPER_M / 0.48) ** 2)
    glass_thickness_m = 0.00608
    glass_thickness_err_m = CALIPER_M

    acrylic_area_m2 = 0.48 * 0.33
    acrylic_area_err_m2 = acrylic_area_m2 * math.sqrt((0.005 / 0.48) ** 2 + (0.01 / 0.33) ** 2)
    acrylic_thickness_m = 0.0070
    acrylic_thickness_err_m = CALIPER_M

    dielectric_results = {
        "Air": dielectric_permittivity(
            air_mean, air_mean_err, air_area_m2, air_thickness_m,
            area_err_m2=air_area_err_m2, thickness_err_m=air_thickness_err_m,
        ),
        "Glass": dielectric_permittivity(
            glass[0], glass[1], glass_area_m2, glass_thickness_m,
            area_err_m2=glass_area_err_m2, thickness_err_m=glass_thickness_err_m,
        ),
        "Acrylic": dielectric_permittivity(
            acrylic[0],
            acrylic[1],
            acrylic_area_m2,
            acrylic_thickness_m,
            area_err_m2=acrylic_area_err_m2,
            thickness_err_m=acrylic_thickness_err_m,
        ),
    }
    air_frequency_dielectric = dielectric_permittivity(
        air_frequency[0],
        air_frequency[1],
        air_area_m2,
        air_thickness_m,
        area_err_m2=air_area_err_m2,
        thickness_err_m=air_thickness_err_m,
    )
    plot_permittivity_summary(
        {label: (values[2], values[3]) for label, values in dielectric_results.items()}
    )

    print("FIT RESULTS")
    for config in RUNS:
        fit = fits[config.key]
        print(
            f"{fit['label']}: "
            f"A={format_pm(float(fit['amplitude_v']), float(fit['amplitude_err_v']), digits=4)} V, "
            f"tau={format_pm(float(fit['tau_s']), float(fit['tau_err_s']), 1e6, 3)} us, "
            f"Voff={format_pm(float(fit['offset_v']), float(fit['offset_err_v']), digits=4)} V, "
            f"sigma_V={float(fit['sigma_v']):.4f} V, "
            f"chi2_red={float(fit['reduced_chi2']):.3f}, "
            f"Ceff={format_pm(float(fit['capacitance_f']), float(fit['capacitance_err_f']), 1e12, 2)} pF"
        )

    print("\nFREQUENCY-RESPONSE FIT RESULTS")
    for config in FREQUENCY_RUNS:
        fit = frequency_fits[config.key]
        print(
            f"{fit['label']}: "
            f"V0={format_pm(float(fit['low_frequency_vpp']), float(fit['low_frequency_err_vpp']), digits=3)} Vpp, "
            f"fc={format_pm(float(fit['corner_hz']), float(fit['corner_err_hz']), 1e-3, 3)} kHz, "
            f"sigma_V={float(fit['sigma_v']):.4f} V, "
            f"chi2_red={float(fit['reduced_chi2']):.3f}, "
            f"Ceff={format_pm(float(fit['capacitance_f']), float(fit['capacitance_err_f']), 1e12, 2)} pF"
        )

    print("\nCORRECTED CAPACITANCES")
    print(f"Air trial 1: {format_pm(air_1[0], air_1[1], 1e12, 2)} pF")
    print(f"Air trial 2: {format_pm(air_2[0], air_2[1], 1e12, 2)} pF")
    print(
        f"Air weighted mean: {format_pm(air_mean, air_mean_err, 1e12, 2)} pF "
        f"(weighted reduced chi2={air_weighted_red_chi2:.3f})"
    )
    print(f"Glass: {format_pm(glass[0], glass[1], 1e12, 2)} pF")
    print(f"Acrylic: {format_pm(acrylic[0], acrylic[1], 1e12, 2)} pF")
    print(f"Air from frequency response: {format_pm(air_frequency[0], air_frequency[1], 1e12, 2)} pF")

    print("\nPERMITTIVITY RESULTS")
    print(
        "Air geometry: "
        f"A={format_pm(air_area_m2, air_area_err_m2, digits=6)} m^2, "
        f"d={format_pm(air_thickness_m, air_thickness_err_m, digits=5)} m"
    )
    print(
        "Glass geometry: "
        f"A={format_pm(glass_area_m2, glass_area_err_m2, digits=5)} m^2, "
        f"d={format_pm(glass_thickness_m, glass_thickness_err_m, digits=6)} m"
    )
    print(
        "Acrylic geometry: "
        f"A={acrylic_area_m2:.4f} +/- {acrylic_area_err_m2:.4f} m^2, "
        f"d={format_pm(acrylic_thickness_m, acrylic_thickness_err_m, digits=5)} m"
    )
    for label, values in dielectric_results.items():
        epsilon, epsilon_err, kappa, kappa_err = values
        print(
            f"{label}: epsilon={format_pm(epsilon, epsilon_err, digits=14)} F/m, "
            f"kappa={format_pm(kappa, kappa_err, digits=2)}"
        )
    print(
        "Air from frequency response: "
        f"epsilon={format_pm(air_frequency_dielectric[0], air_frequency_dielectric[1], digits=14)} F/m, "
        f"kappa={format_pm(air_frequency_dielectric[2], air_frequency_dielectric[3], digits=2)}"
    )


if __name__ == "__main__":
    main()