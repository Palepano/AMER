import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from matplotlib.lines import Line2D

# ============================================================
# АВТОПОЭТИЧЕСКАЯ МАТРИЧНАЯ МОДЕЛЬ — AMER / CAUSAL v23
# ------------------------------------------------------------
# Цель версии v23:
#   - сохранить форму циклов версии 13/14/15;
#   - выполнить локальную доводку только в точках пересечения с линией баланса, особенно на спаде;
#   - сделать линии баланса по панелям структурно различающимися, но полностью эндогенными;
#   - полностью увязать абсолютные уровни 1-й и 2-й панелей с инвестиционными рядами 3-й и 4-й панелей;
#   - сохранить почти постоянную, но более выраженную амплитуду цикла в абсолютных величинах при восходящем тренде, с очень слабой эндогенной модуляцией от структуры накопленного капитала.
# ============================================================

BALANCE_PERCENT = 30.0
DISPLAY_CYCLES = 3
DISPLAY_CYCLE_LEN = 260
INTERNAL_T = 9000
OUTDIR = os.path.abspath(os.getenv("AMER_OUTDIR", "/content" if os.path.exists("/content") else "."))
OUT_PNG = os.path.join(OUTDIR, "amer_matrix_v23_final_touch_balance_crossings_preview.png")
OUT_CSV = os.path.join(OUTDIR, "amer_matrix_v23_final_touch_balance_crossings_data.csv")
FIGSIZE = (16, 14)


@dataclass
class AMERConfig:
    # Производственная матрица: продукты x ресурсы
    A: np.ndarray = field(default_factory=lambda: np.array([
        [0.52, 0.10, 0.12],
        [0.36, 0.24, 0.11],
        [0.22, 0.34, 0.20],
    ], dtype=float))

    # Воспроизводственная матрица: ресурсы x продукты
    B: np.ndarray = field(default_factory=lambda: np.array([
        [0.58, 0.09, 0.03],
        [0.34, 0.22, 0.05],
        [0.48, 0.11, 0.09],
    ], dtype=float))

    # Нелинейный эндогенный генератор
    T: int = INTERNAL_T
    dt: float = 0.03
    mu: float = 4.6
    x0: float = -1.8
    y0: float = 0.15

    # Геометрия среднего цикла
    target_up_cross: float = 0.31
    target_peak: float = 0.65
    target_down_cross: float = 0.775

    # Амплитуды разрывов выше линии баланса
    ds_gap_amp: float = 13.8      # AS - AD
    ws_gap_amp: float = 10.4       # Sw - Pw
    inv_gap_amp: float = 8.4      # IKT - IHT = GNI - GDP

    # Сглаживание линии баланса
    balance_band: float = 8.0
    balance_sigma: float = 5.0
    cross_time_width: float = 12.0
    peak_width: float = 24.0
    base_smoothing_passes: int = 4
    gap_smoothing_passes: int = 3

    # Вес канонической фазовой формы поверх эндогенного цикла
    canonical_weight: float = 0.72

    # Геометрия подъема и спада с согласованием наклонов в точках стыка
    rise_balance_phys_slope: float = 0.80    # наклон в точке линии баланса, % на шаг
    rise_peak_phys_slope: float = 0.80       # наклон возле вершины
    descent_peak_phys_slope: float = -2.08   # старт спада от пика
    descent_balance_phys_slope: float = -1.98  # наклон у линии баланса на спаде
    tail_floor_phys_slope: float = -0.03     # у дна почти горизонтально
    upper_linear_weight_rise: float = 0.95
    upper_linear_weight_descent: float = 0.985
    upper_segment_enforce: float = 0.74
    upper_straighten_rise: float = 0.34
    upper_straighten_descent: float = 0.44

    # Параметры очень плавного расхождения выше линии баланса
    asym_rise_power: float = 3.10
    asym_level_power: float = 1.90
    asym_gap_power: float = 1.45
    merge_early_point: float = 0.31
    merge_decay_power: float = 1.12

    # Лаги и инерции: чем меньше alpha, тем сильнее отставание
    alpha_fast_up: float = 0.36
    alpha_fast_down: float = 0.52
    alpha_mid_up: float = 0.26
    alpha_mid_down: float = 0.43
    alpha_slow_up: float = 0.18
    alpha_slow_down: float = 0.37
    alpha_crash: float = 0.82
    peak_exit_gain_fast: float = 0.18
    peak_exit_gain_mid: float = 0.25
    peak_exit_gain_slow: float = 0.34
    peak_exit_sigma: float = 4.0
    peak_exit_relax: float = 8.4

    # Скрытые абсолютные потоки
    created_base: float = 1.00
    created_scale: float = 5.20
    added_share_max: float = 0.36



def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def normalize_0_100(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    denom = max(float(s.max() - s.min()), 1e-9)
    return 100.0 * (s - s.min()) / denom


def smoothstep(x, edge0, edge1):
    x = np.asarray(x, dtype=float)
    if edge1 == edge0:
        return (x >= edge1).astype(float)
    u = clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_series(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    if passes <= 0:
        return x
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    for _ in range(passes):
        pad = np.pad(x, (2, 2), mode="edge")
        x = np.convolve(pad, kernel, mode="same")[2:-2]
    return x


def detect_troughs(series: np.ndarray, min_gap: int = 60):
    s = np.asarray(series, dtype=float)
    troughs = []
    for i in range(1, len(s) - 1):
        if s[i - 1] > s[i] <= s[i + 1]:
            if not troughs or i - troughs[-1] >= min_gap:
                troughs.append(i)
    return troughs


def phase_points(series_pct: np.ndarray, balance: float = BALANCE_PERCENT):
    y = np.asarray(series_pct, dtype=float)
    peak = int(np.argmax(y))

    up_cross = None
    down_cross = None

    for i in range(1, peak + 1):
        if y[i - 1] < balance <= y[i]:
            up_cross = i
            break

    for i in range(peak + 1, len(y)):
        if y[i - 1] > balance >= y[i]:
            down_cross = i
            break

    if up_cross is None:
        up_cross = max(1, int(0.30 * (len(y) - 1)))
    if down_cross is None:
        down_cross = min(len(y) - 2, peak + max(2, int(0.13 * len(y))))

    return {
        "trough_start": 0,
        "up_cross": up_cross,
        "peak": peak,
        "down_cross": down_cross,
        "trough_end": len(y) - 1,
    }


def warp_cycle(series: np.ndarray, phase_src: dict, n_out: int, cfg: AMERConfig):
    src = np.arange(len(series), dtype=float)
    dst = np.arange(n_out, dtype=float)

    phase_dst = {
        "trough_start": 0,
        "up_cross": int(round(cfg.target_up_cross * (n_out - 1))),
        "peak": int(round(cfg.target_peak * (n_out - 1))),
        "down_cross": int(round(cfg.target_down_cross * (n_out - 1))),
        "trough_end": n_out - 1,
    }

    src_pts = np.array([
        phase_src["trough_start"],
        phase_src["up_cross"],
        phase_src["peak"],
        phase_src["down_cross"],
        phase_src["trough_end"],
    ], dtype=float)
    dst_pts = np.array([
        phase_dst["trough_start"],
        phase_dst["up_cross"],
        phase_dst["peak"],
        phase_dst["down_cross"],
        phase_dst["trough_end"],
    ], dtype=float)

    src_of_dst = np.interp(dst, dst_pts, src_pts)
    warped = np.interp(src_of_dst, src, np.asarray(series, dtype=float))
    return warped, phase_dst


def simulate_endogenous_generator(cfg: AMERConfig):
    x = np.zeros(cfg.T, dtype=float)
    y = np.zeros(cfg.T, dtype=float)
    x[0] = cfg.x0
    y[0] = cfg.y0

    for t in range(cfg.T - 1):
        dx = y[t]
        dy = cfg.mu * (1.0 - x[t] ** 2) * y[t] - x[t]
        x[t + 1] = x[t] + cfg.dt * dx
        y[t + 1] = y[t] + cfg.dt * dy

    return x, y


def bell(idx, center, width):
    width = max(float(width), 1e-6)
    return np.exp(-0.5 * ((idx - center) / width) ** 2)


def cubic_hermite(y0, y1, m0, m1, u):
    u = clip(np.asarray(u, dtype=float), 0.0, 1.0)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def canonical_cycle_profile(n: int, phase: dict, cfg: AMERConfig):
    idx = np.arange(n, dtype=float)
    y = np.zeros(n, dtype=float)

    a = phase["trough_start"]
    b = phase["up_cross"]
    c = phase["peak"]
    d = phase["down_cross"]
    e = phase["trough_end"]

    # 1) От дна до линии баланса: подъем с ненулевым наклоном к линии баланса.
    if b > a:
        L1 = max(b - a, 1)
        m0_low = 0.0
        m1_low = cfg.rise_balance_phys_slope * L1 / max(BALANCE_PERCENT, 1e-6)
        u = (idx[a:b + 1] - a) / L1
        seg = cubic_hermite(0.0, 1.0, m0_low, m1_low, u)
        y[a:b + 1] = BALANCE_PERCENT * clip(seg, 0.0, 1.0)

    # 2) От линии баланса до пика: еще более прямой подъем, но без излома в b.
    if c > b:
        L2 = max(c - b, 1)
        m0_high = cfg.rise_balance_phys_slope * L2 / max(100.0 - BALANCE_PERCENT, 1e-6)
        m1_high = cfg.rise_peak_phys_slope * L2 / max(100.0 - BALANCE_PERCENT, 1e-6)
        u = (idx[b:c + 1] - b) / L2
        herm = cubic_hermite(0.0, 1.0, m0_high, m1_high, u)
        lin = u
        seg = (1.0 - cfg.upper_linear_weight_rise) * herm + cfg.upper_linear_weight_rise * lin
        y[b:c + 1] = BALANCE_PERCENT + (100.0 - BALANCE_PERCENT) * clip(seg, 0.0, 1.0)

    # 3) От пика к линии баланса: почти прямой резкий спад, но без излома.
    if d > c:
        L3 = max(d - c, 1)
        m0_drop = cfg.descent_peak_phys_slope * L3 / max(100.0 - BALANCE_PERCENT, 1e-6)
        m1_drop = cfg.descent_balance_phys_slope * L3 / max(100.0 - BALANCE_PERCENT, 1e-6)
        u = (idx[c:d + 1] - c) / L3
        herm = cubic_hermite(1.0, 0.0, m0_drop, m1_drop, u)
        lin = 1.0 - u
        seg = (1.0 - cfg.upper_linear_weight_descent) * herm + cfg.upper_linear_weight_descent * lin
        y[c:d + 1] = BALANCE_PERCENT + (100.0 - BALANCE_PERCENT) * clip(seg, 0.0, 1.0)

    # 4) Ниже линии баланса: продолжение спада без излома в d и плавный выход к дну.
    if e > d:
        L4 = max(e - d, 1)
        m0_tail = cfg.descent_balance_phys_slope * L4 / max(BALANCE_PERCENT, 1e-6)
        m1_tail = cfg.tail_floor_phys_slope * L4 / max(BALANCE_PERCENT, 1e-6)
        u = (idx[d:e + 1] - d) / L4
        seg = cubic_hermite(1.0, 0.0, m0_tail, m1_tail, u)
        y[d:e + 1] = BALANCE_PERCENT * clip(seg, 0.0, 1.0)

    y = clip(y, 0.0, 100.0)
    return y




def linear_upper_reference(n: int, phase: dict):
    idx = np.arange(n, dtype=float)
    y = np.full(n, np.nan, dtype=float)

    b = phase["up_cross"]
    c = phase["peak"]
    d = phase["down_cross"]

    if c > b:
        u = (idx[b:c + 1] - b) / max(c - b, 1)
        y[b:c + 1] = BALANCE_PERCENT + (100.0 - BALANCE_PERCENT) * u

    if d > c:
        u = (idx[c:d + 1] - c) / max(d - c, 1)
        y[c:d + 1] = 100.0 - (100.0 - BALANCE_PERCENT) * u

    return y


def straighten_upper_segments(series: np.ndarray, phase: dict, cfg: AMERConfig):
    idx = np.arange(len(series), dtype=float)
    ref = linear_upper_reference(len(series), phase)
    out = np.asarray(series, dtype=float).copy()

    b = phase["up_cross"]
    c = phase["peak"]
    d = phase["down_cross"]

    rise_gate = smoothstep(idx, b - 3.0, b + 8.0) * (1.0 - smoothstep(idx, c - 6.0, c + 3.0))
    drop_gate = smoothstep(idx, c - 2.0, c + 4.0) * (1.0 - smoothstep(idx, d - 4.0, d + 5.0))

    rise_mask = ~np.isnan(ref)
    if np.any(rise_mask):
        wr = cfg.upper_straighten_rise * rise_gate
        wd = cfg.upper_straighten_descent * drop_gate
        w = clip(wr + wd, 0.0, 0.90)
        out[rise_mask] = (1.0 - w[rise_mask]) * out[rise_mask] + w[rise_mask] * ref[rise_mask]

    out = smooth_series(out, passes=1)
    out = clip(out, 0.0, 100.0)
    return out



def micro_smooth_joints(series: np.ndarray, phase: dict, radius: int = 4) -> np.ndarray:
    x = np.asarray(series, dtype=float).copy()
    y1 = smooth_series(x, passes=1)
    y2 = smooth_series(x, passes=2)

    up = int(phase["up_cross"])
    down = int(phase["down_cross"])

    lo = max(0, up - radius)
    hi = min(len(x), up + radius + 1)
    x[lo:hi] = 0.72 * y1[lo:hi] + 0.28 * y2[lo:hi]

    down_radius = radius + 2
    lo = max(0, down - down_radius)
    hi = min(len(x), down + down_radius + 1)
    x[lo:hi] = 0.42 * x[lo:hi] + 0.33 * y1[lo:hi] + 0.25 * y2[lo:hi]
    return x

def blend_endogenous_with_canonical(base_pct: np.ndarray, cfg: AMERConfig):
    phase = phase_points(base_pct, BALANCE_PERCENT)
    canonical = canonical_cycle_profile(len(base_pct), phase, cfg)

    idx = np.arange(len(base_pct), dtype=float)
    upper_plateau = smoothstep(idx, phase["up_cross"] - 10.0, phase["up_cross"] + 10.0)
    upper_plateau *= 1.0 - smoothstep(idx, phase["down_cross"] - 10.0, phase["down_cross"] + 10.0)

    weight_window = (
        0.30 * bell(idx, phase["up_cross"], 24.0)
        + 0.54 * bell(idx, phase["peak"], 30.0)
        + 0.46 * bell(idx, phase["down_cross"], 20.0)
        + cfg.upper_segment_enforce * upper_plateau
    )
    weight_window = clip(weight_window, 0.0, 1.0)
    w = cfg.canonical_weight * weight_window + 0.18

    shaped = (1.0 - w) * base_pct + w * canonical
    shaped = straighten_upper_segments(shaped, phase, cfg)
    shaped = micro_smooth_joints(shaped, phase, radius=4)
    shaped = normalize_0_100(shaped)
    shaped = straighten_upper_segments(shaped, phase_points(shaped, BALANCE_PERCENT), cfg)
    return shaped


def adaptive_follow(target, asym_eff, idx, phase, cfg: AMERConfig, role: str):
    target = np.asarray(target, dtype=float)
    asym_eff = np.asarray(asym_eff, dtype=float)
    out = np.zeros_like(target)
    out[0] = target[0]

    peak = phase["peak"]
    down_cross = phase["down_cross"]

    crash_gate = sigmoid((idx - (peak + 1.3)) / max(cfg.peak_exit_sigma, 1e-6))
    crash_relax = 1.0 - sigmoid((idx - (down_cross - 5.0)) / max(cfg.peak_exit_relax, 1e-6))
    peak_exit_wave = clip(crash_gate * crash_relax, 0.0, 1.0)

    for i in range(1, len(target)):
        delta = target[i] - out[i - 1]
        boom = float(asym_eff[i])
        exit_wave = float(peak_exit_wave[i])

        if role == "fast":
            a_up = cfg.alpha_fast_up + 0.03 * boom
            a_down = cfg.alpha_fast_down + 0.04 * exit_wave
            exit_gain = cfg.peak_exit_gain_fast
        elif role == "mid":
            a_up = cfg.alpha_mid_up + 0.03 * boom
            a_down = cfg.alpha_mid_down + 0.08 * exit_wave
            exit_gain = cfg.peak_exit_gain_mid
        elif role == "slow":
            a_up = cfg.alpha_slow_up - 0.01 * boom
            a_down = cfg.alpha_slow_down + 0.11 * exit_wave
            exit_gain = cfg.peak_exit_gain_slow
        elif role == "demand":
            a_up = cfg.alpha_slow_up - 0.02 * boom
            a_down = cfg.alpha_slow_down + 0.13 * exit_wave
            exit_gain = cfg.peak_exit_gain_slow
        elif role == "human":
            a_up = cfg.alpha_slow_up - 0.02 * boom
            a_down = cfg.alpha_slow_down + 0.12 * exit_wave
            exit_gain = cfg.peak_exit_gain_slow
        else:
            a_up = cfg.alpha_mid_up
            a_down = cfg.alpha_mid_down
            exit_gain = cfg.peak_exit_gain_mid

        a_up = clip(a_up, 0.05, 0.85)
        a_down = clip(a_down, 0.05, 0.96)

        alpha = a_up if delta >= 0.0 else a_down
        if delta < 0.0:
            alpha = clip(alpha + exit_gain * exit_wave, 0.05, cfg.alpha_crash)

        out[i] = out[i - 1] + alpha * delta
    return out


def build_cycle_frame(base_pct: np.ndarray, t: np.ndarray, cycle_no: int, cfg: AMERConfig):
    phase = phase_points(base_pct, BALANCE_PERCENT)
    idx = np.arange(len(base_pct), dtype=float)
    peak = phase["peak"]
    up_cross = phase["up_cross"]
    down_cross = phase["down_cross"]

    # Очень плавное нарушение симметрии выше линии баланса.
    rel_above = clip((base_pct - BALANCE_PERCENT) / max(100.0 - BALANCE_PERCENT, 1e-9), 0.0, 1.0)
    level_gate = smoothstep(rel_above, 0.0, 1.0) ** cfg.asym_level_power

    rise_u = clip((idx - up_cross) / max(peak - up_cross, 1), 0.0, 1.0)
    rise_gate = smoothstep(rise_u, 0.0, 1.0) ** cfg.asym_rise_power

    decay_u = clip((idx - peak) / max(down_cross - peak, 1), 0.0, 1.0)
    decay_gate = 1.0 - smoothstep(decay_u, 0.0, cfg.merge_early_point) ** cfg.merge_decay_power

    peak_boost = 0.92 + 0.08 * bell(idx, peak, cfg.peak_width)

    asym_eff = clip(level_gate * rise_gate * decay_gate * peak_boost, 0.0, 1.0)
    asym_eff = smooth_series(asym_eff, passes=cfg.gap_smoothing_passes)
    asym_eff = clip(asym_eff, 0.0, 1.0)

    ds_gap = smooth_series(cfg.ds_gap_amp * (asym_eff ** cfg.asym_gap_power), passes=cfg.gap_smoothing_passes)
    ws_gap = smooth_series(cfg.ws_gap_amp * (asym_eff ** cfg.asym_gap_power), passes=cfg.gap_smoothing_passes)
    inv_gap = smooth_series(cfg.inv_gap_amp * (asym_eff ** cfg.asym_gap_power), passes=cfg.gap_smoothing_passes)

    target_GDP = clip(base_pct - 0.5 * inv_gap, 0.0, 100.0)
    target_GNI = clip(base_pct + 0.5 * inv_gap, 0.0, 100.0)

    target_AD = clip(base_pct - 0.5 * ds_gap, 0.0, 100.0)
    target_AS = clip(base_pct + 0.5 * ds_gap, 0.0, 100.0)

    target_Pw = clip(base_pct - 0.5 * ws_gap, 0.0, 100.0)
    target_Sw = clip(base_pct + 0.5 * ws_gap, 0.0, 100.0)

    target_IKT = clip(base_pct + 0.5 * inv_gap, 0.0, 100.0)
    target_IHT = clip(base_pct - 0.5 * inv_gap, 0.0, 100.0)

    GNI_pct = adaptive_follow(target_GNI, asym_eff, idx, phase, cfg, role="fast")
    GDP_pct = adaptive_follow(target_GDP, asym_eff, idx, phase, cfg, role="mid")

    AS_pct = adaptive_follow(target_AS, asym_eff, idx, phase, cfg, role="fast")
    AD_pct = adaptive_follow(target_AD, asym_eff, idx, phase, cfg, role="demand")

    Pw_pct = adaptive_follow(target_Pw, asym_eff, idx, phase, cfg, role="mid")
    Sw_pct = adaptive_follow(target_Sw, asym_eff, idx, phase, cfg, role="fast")

    IKT_pct = adaptive_follow(target_IKT, asym_eff, idx, phase, cfg, role="fast")
    IHT_pct = adaptive_follow(target_IHT, asym_eff, idx, phase, cfg, role="human")

    GDP_pct = clip(smooth_series(GDP_pct, passes=1), 0.0, 100.0)
    GNI_pct = clip(smooth_series(GNI_pct, passes=1), 0.0, 100.0)
    AD_pct = clip(smooth_series(AD_pct, passes=1), 0.0, 100.0)
    AS_pct = clip(smooth_series(AS_pct, passes=1), 0.0, 100.0)
    Pw_pct = clip(smooth_series(Pw_pct, passes=1), 0.0, 100.0)
    Sw_pct = clip(smooth_series(Sw_pct, passes=1), 0.0, 100.0)
    IKT_pct = clip(smooth_series(IKT_pct, passes=1), 0.0, 100.0)
    IHT_pct = clip(smooth_series(IHT_pct, passes=1), 0.0, 100.0)

    # Скрытые абсолютные потоки.
    created = cfg.created_base + cfg.created_scale * (base_pct / 100.0) ** 1.05
    Pc = created.copy()
    Sc = created.copy()
    Mc = created.copy()

    net_base = 0.16 + 0.62 * (base_pct / 100.0)
    P_net = net_base * (1.0 + 0.36 * asym_eff)
    S_net = net_base * (1.0 - 0.26 * asym_eff)

    Pw = np.maximum(Pc - P_net, 1e-6)
    Sw = np.maximum(Sc - S_net, 1e-6)

    added_share = cfg.added_share_max * asym_eff
    P_A = Pc * added_share
    P_N = Pc - P_A

    S_A = Sc * added_share
    S_N = Sc - S_A

    M_A = Mc * added_share
    M_N = Mc - M_A

    B_A = created * added_share
    B_N = created - B_A

    I_H_N = Pw.copy()
    I_K_N = Sw.copy()

    R_K_N = 0.34 * P_net
    R_H_N = 0.34 * S_net

    I_K_T_abs = 0.8 + 5.1 * (IKT_pct / 100.0)
    I_H_T_abs = 0.8 + 5.1 * (IHT_pct / 100.0)

    rest_K = np.maximum(I_K_T_abs - I_K_N - R_K_N, 0.0)
    rest_H = np.maximum(I_H_T_abs - I_H_N - R_H_N, 0.0)

    I_K_A = 0.55 * rest_K
    R_K_A = 0.45 * rest_K
    I_H_A = 0.55 * rest_H
    R_H_A = 0.45 * rest_H

    I_K_T_abs = I_K_N + I_K_A + R_K_N + R_K_A
    I_H_T_abs = I_H_N + I_H_A + R_H_N + R_H_A

    P_net_check = Pc - Pw
    S_net_check = Sc - Sw
    I_K_T_check = I_K_N + I_K_A + R_K_N + R_K_A
    I_H_T_check = I_H_N + I_H_A + R_H_N + R_H_A

    frame = pd.DataFrame({
        "t": t,
        "cycle_no": cycle_no,
        "base_pct": base_pct,
        "GDP_pct": GDP_pct,
        "GNI_pct": GNI_pct,
        "AD_pct": AD_pct,
        "AS_pct": AS_pct,
        "Pw_pct": Pw_pct,
        "Sw_pct": Sw_pct,
        "IKT_pct": IKT_pct,
        "IHT_pct": IHT_pct,
        "target_GDP": target_GDP,
        "target_GNI": target_GNI,
        "target_AD": target_AD,
        "target_AS": target_AS,
        "asymmetry": asym_eff,
        "Pc": Pc,
        "Pw": Pw,
        "P_net": P_net,
        "P_net_check": P_net_check,
        "P_N": P_N,
        "P_A": P_A,
        "Sc": Sc,
        "Sw": Sw,
        "S_net": S_net,
        "S_net_check": S_net_check,
        "S_N": S_N,
        "S_A": S_A,
        "Mc": Mc,
        "M_N": M_N,
        "M_A": M_A,
        "B_N": B_N,
        "B_A": B_A,
        "I_K_N": I_K_N,
        "R_K_N": R_K_N,
        "I_K_A": I_K_A,
        "R_K_A": R_K_A,
        "I_H_N": I_H_N,
        "R_H_N": R_H_N,
        "I_H_A": I_H_A,
        "R_H_A": R_H_A,
        "I_K_T_abs": I_K_T_abs,
        "I_H_T_abs": I_H_T_abs,
        "I_K_T_check": I_K_T_check,
        "I_H_T_check": I_H_T_check,
    })
    return frame, phase


def build_display_data(cfg: AMERConfig):
    x, y = simulate_endogenous_generator(cfg)
    troughs = detect_troughs(x, min_gap=80)
    if len(troughs) < DISPLAY_CYCLES + 4:
        raise RuntimeError("Не удалось обнаружить достаточное число эндогенных циклов.")

    start_idx = 5
    chosen = [(troughs[i], troughs[i + 1]) for i in range(start_idx, start_idx + DISPLAY_CYCLES)]

    frames = []
    events = []
    offset = 0

    for cycle_no, (a, b) in enumerate(chosen, start=1):
        raw = x[a:b + 1]
        raw_pct = normalize_0_100(raw)
        src_phase = phase_points(raw_pct, BALANCE_PERCENT)
        base_pct, dst_phase = warp_cycle(raw_pct, src_phase, DISPLAY_CYCLE_LEN, cfg)

        base_pct = smooth_series(base_pct, passes=cfg.base_smoothing_passes)
        base_pct = normalize_0_100(base_pct)
        base_pct = blend_endogenous_with_canonical(base_pct, cfg)
        dst_phase = phase_points(base_pct, BALANCE_PERCENT)
        base_pct = micro_smooth_joints(base_pct, dst_phase, radius=4)
        base_pct = straighten_upper_segments(base_pct, dst_phase, cfg)
        base_pct = normalize_0_100(base_pct)
        dst_phase = phase_points(base_pct, BALANCE_PERCENT)

        t = np.arange(offset, offset + DISPLAY_CYCLE_LEN)
        frame, _ = build_cycle_frame(base_pct, t, cycle_no, cfg)
        frames.append(frame)

        events.append({k: offset + v for k, v in dst_phase.items()})
        offset += DISPLAY_CYCLE_LEN

    df = pd.concat(frames, ignore_index=True)
    return df, events, chosen


def draw_phase_lines(ax, events):
    for ev in events:
        ax.axvline(ev["trough_start"], color="gray", linestyle=":", linewidth=1.0, alpha=0.62)
        ax.axvline(ev["up_cross"], color="gray", linestyle="--", linewidth=1.05, alpha=0.82)
        ax.axvline(ev["peak"], color="gray", linestyle="-.", linewidth=1.15, alpha=0.90)
        ax.axvline(ev["down_cross"], color="gray", linestyle="--", linewidth=1.05, alpha=0.82)
        ax.axvline(ev["trough_end"], color="gray", linestyle=":", linewidth=1.0, alpha=0.62)


def plot_four_panels(df: pd.DataFrame, events, out_path: str = OUT_PNG):
    ensure_parent_dir(out_path)
    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"hspace": 0.10})

    for ax in axes:
        draw_phase_lines(ax, events)
        ax.axhline(BALANCE_PERCENT, color="gray", linestyle=(0, (6, 5)), linewidth=1.4, alpha=0.9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("%")

    axes[0].plot(df["t"], df["GDP_pct"], linewidth=2.0, label="ВВП")
    axes[0].plot(df["t"], df["GNI_pct"], linewidth=2.0, label="ВНД")
    axes[0].set_title("ВВП / ВНД")
    phase_handles = [
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.0, label="дно цикла"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.05, label="пересечение линии баланса"),
        Line2D([0], [0], color="gray", linestyle="-.", linewidth=1.15, label="пик цикла"),
    ]
    h0, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h0 + phase_handles, loc="upper left", ncol=2)

    axes[1].plot(df["t"], df["AS_pct"], linewidth=2.0, label="предложение")
    axes[1].plot(df["t"], df["AD_pct"], linewidth=2.0, label="спрос")
    axes[1].set_title("Спрос / предложение")
    axes[1].legend(loc="upper left")

    axes[2].plot(df["t"], df["Pw_pct"], linewidth=2.0, label="изымаемая прибыль")
    axes[2].plot(df["t"], df["Sw_pct"], linewidth=2.0, label="изымаемые сбережения")
    axes[2].set_title("Изымаемая прибыль / Изымаемые сбережения")
    axes[2].legend(loc="upper left")

    axes[3].plot(df["t"], df["IKT_pct"], linewidth=2.0, label="совокупные инвестиции в физический капитал")
    axes[3].plot(df["t"], df["IHT_pct"], linewidth=2.0, label="совокупные инвестиции в человеческий капитал")
    axes[3].set_title("Совокупные инвестиции в физический капитал / человеческий капитал")
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Количество итераций")

    fig.suptitle(
        "Эндогенная саморегулируемая матричная модель | микродоводка верхних участков | гладкие стыки с балансом",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)



def capital_accumulation_paths(df: pd.DataFrame):
    # Необходимые вложения поддерживают уже имеющийся капитал,
    # а добавочные вложения и реинвестиции наращивают его.
    net_add_K = df["I_K_A"].to_numpy(dtype=float) + df["R_K_A"].to_numpy(dtype=float)
    net_add_H = df["I_H_A"].to_numpy(dtype=float) + df["R_H_A"].to_numpy(dtype=float)

    maint_K0 = float(df["I_K_N"].iloc[0] + df["R_K_N"].iloc[0])
    maint_H0 = float(df["I_H_N"].iloc[0] + df["R_H_N"].iloc[0])

    K0 = 12.0 * maint_K0
    H0 = 12.0 * maint_H0

    K_stock = K0 + np.cumsum(net_add_K)
    H_stock = H0 + np.cumsum(net_add_H)

    K_stock = np.maximum.accumulate(K_stock)
    H_stock = np.maximum.accumulate(H_stock)

    K_stock = smooth_series(K_stock, passes=2)
    H_stock = smooth_series(H_stock, passes=2)

    # Панельные драйверы линии баланса: все эндогенны, но структурно различаются.
    # 1) Выпуск и доход сильнее зависят от физического капитала.
    bal1_driver = (K_stock ** 0.62) * (H_stock ** 0.38)
    # 2) Спрос и предложение немного сильнее опираются на человеческий капитал.
    bal2_driver = (K_stock ** 0.48) * (H_stock ** 0.52)
    # 3) Прибыль/сбережения зависят от обеих форм капитала почти симметрично.
    bal3_driver = (K_stock ** 0.50) * (H_stock ** 0.50)
    # 4) Инвестиционная панель опять немного сильнее тяготеет к физическому капиталу.
    bal4_driver = (K_stock ** 0.58) * (H_stock ** 0.42)

    bal1_driver = np.maximum.accumulate(smooth_series(bal1_driver, passes=2))
    bal2_driver = np.maximum.accumulate(smooth_series(bal2_driver, passes=2))
    bal3_driver = np.maximum.accumulate(smooth_series(bal3_driver, passes=2))
    bal4_driver = np.maximum.accumulate(smooth_series(bal4_driver, passes=2))

    return K_stock, H_stock, bal1_driver, bal2_driver, bal3_driver, bal4_driver



def pct_to_absolute_weakly_modulated_amplitude(
    pct: np.ndarray,
    balance_line: np.ndarray,
    up_amp_abs: float,
    down_amp_abs: float,
    modulation_signal: np.ndarray,
    modulation_strength: float = 0.045,
    ref_balance: float | None = None,
    smooth_width_pct: float = 3.0,
) -> np.ndarray:
    """Перевод процентов вокруг линии баланса в абсолютные уровни так,
    чтобы амплитуда оставалась почти постоянной, но с очень слабой эндогенной
    модуляцией от структуры накопленного капитала.

    Главное отличие v20: вблизи линии баланса переход между верхней и нижней
    полуамплитудой делается не кусочно-линейным, а мягко-сигмоидальным.
    Это убирает остаточный излом в точке пересечения, особенно на спаде.
    """
    pct = np.asarray(pct, dtype=float)
    bal = np.asarray(balance_line, dtype=float)
    mod = np.asarray(modulation_signal, dtype=float)

    if ref_balance is None:
        ref_balance = float(bal[0])
    ref_balance = max(ref_balance, 1e-9)

    mod = smooth_series(mod, passes=2)
    mod = mod - float(mod[0])
    mod_scale = max(float(np.max(np.abs(mod))), 1e-9)
    mod = mod / mod_scale
    amp_mod = 1.0 + modulation_strength * mod
    amp_mod = clip(amp_mod, 1.0 - 1.6 * modulation_strength, 1.0 + 1.6 * modulation_strength)

    # Мягкий переход через линию баланса: одновременно сглаживаются
    # и амплитуда, и нормировочный знаменатель.
    trans = sigmoid((pct - BALANCE_PERCENT) / max(smooth_width_pct, 1e-9))
    amp = down_amp_abs + (up_amp_abs - down_amp_abs) * trans
    denom = BALANCE_PERCENT + ((100.0 - BALANCE_PERCENT) - BALANCE_PERCENT) * trans
    signed_rel = (pct - BALANCE_PERCENT) / np.maximum(denom, 1e-9)

    out = bal + ref_balance * amp_mod * amp * signed_rel
    floor = np.maximum(0.12 * ref_balance, 0.08 * bal)
    return np.maximum(out, floor)


def enrich_with_panel_endogenous_weakly_modulated_amplitude(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    K_stock, H_stock, bal1_driver, bal2_driver, bal3_driver, bal4_driver = capital_accumulation_paths(out)
    out["K_stock"] = K_stock
    out["H_stock"] = H_stock
    out["bal1_driver"] = bal1_driver
    out["bal2_driver"] = bal2_driver
    out["bal3_driver"] = bal3_driver
    out["bal4_driver"] = bal4_driver

    # Начальные уровни по панелям.
    base1 = 118.0
    base2 = 112.0
    base3 = 46.0
    base4 = 52.0

    # Нормировка драйверов к стартовому уровню: восходящий тренд полностью эндогенен.
    d1 = bal1_driver / max(float(bal1_driver[0]), 1e-9)
    d2 = bal2_driver / max(float(bal2_driver[0]), 1e-9)
    d3 = bal3_driver / max(float(bal3_driver[0]), 1e-9)
    d4 = bal4_driver / max(float(bal4_driver[0]), 1e-9)

    out["BAL_1_abs"] = base1 * d1
    out["BAL_2_abs"] = base2 * d2
    out["BAL_3_abs"] = base3 * d3
    out["BAL_4_abs"] = base4 * d4

    # Очень слабая эндогенная модуляция амплитуды от структуры накопленного капитала.
    # Если доля физического капитала в совокупном капитале растет относительно человеческого,
    # амплитуда слегка усиливается; если баланс смещается в сторону человеческого капитала —
    # слегка ослабляется. Эффект заведомо мал, чтобы амплитуда оставалась почти постоянной.
    capital_total = np.maximum(K_stock + H_stock, 1e-9)
    k_share = K_stock / capital_total
    h_share = H_stock / capital_total
    structure_signal = smooth_series((k_share - h_share), passes=2)

    # Панельные сигналы модуляции различаются по структуре.
    mod1 = smooth_series(0.75 * structure_signal + 0.25 * np.gradient(np.log(np.maximum(bal1_driver, 1e-9))), passes=2)
    mod2 = smooth_series(0.35 * structure_signal + 0.65 * np.gradient(np.log(np.maximum(bal2_driver, 1e-9))), passes=2)
    mod3 = smooth_series(0.50 * structure_signal + 0.50 * np.gradient(np.log(np.maximum(bal3_driver, 1e-9))), passes=2)
    mod4 = smooth_series(0.80 * structure_signal + 0.20 * np.gradient(np.log(np.maximum(bal4_driver, 1e-9))), passes=2)

    out["amp_mod_1"] = mod1
    out["amp_mod_2"] = mod2
    out["amp_mod_3"] = mod3
    out["amp_mod_4"] = mod4

    ref1 = float(out["BAL_1_abs"].iloc[0])
    ref2 = float(out["BAL_2_abs"].iloc[0])
    ref3 = float(out["BAL_3_abs"].iloc[0])
    ref4 = float(out["BAL_4_abs"].iloc[0])

    # В версии 19 сохраняем верхнюю полуамплитуду версии 18,
    # но усиливаем только нижнюю полуамплитуду, чтобы дно цикла
    # выражалось отчетливее без потери уже достигнутой геометрии. 
    out["GDP_abs"] = pct_to_absolute_weakly_modulated_amplitude(
        out["GDP_pct"], out["BAL_1_abs"], up_amp_abs=2.46, down_amp_abs=1.62,
        modulation_signal=mod1, modulation_strength=0.042, ref_balance=ref1
    )
    out["GNI_abs"] = pct_to_absolute_weakly_modulated_amplitude(
        out["GNI_pct"], out["BAL_1_abs"], up_amp_abs=2.46, down_amp_abs=1.62,
        modulation_signal=mod1, modulation_strength=0.042, ref_balance=ref1
    )

    out["AD_abs"] = pct_to_absolute_weakly_modulated_amplitude(
        out["AD_pct"], out["BAL_2_abs"], up_amp_abs=2.56, down_amp_abs=1.70,
        modulation_signal=mod2, modulation_strength=0.044, ref_balance=ref2
    )
    out["AS_abs"] = pct_to_absolute_weakly_modulated_amplitude(
        out["AS_pct"], out["BAL_2_abs"], up_amp_abs=2.56, down_amp_abs=1.70,
        modulation_signal=mod2, modulation_strength=0.044, ref_balance=ref2
    )

    out["Pw_abs_plot"] = pct_to_absolute_weakly_modulated_amplitude(
        out["Pw_pct"], out["BAL_3_abs"], up_amp_abs=2.02, down_amp_abs=1.48,
        modulation_signal=mod3, modulation_strength=0.040, ref_balance=ref3
    )
    out["Sw_abs_plot"] = pct_to_absolute_weakly_modulated_amplitude(
        out["Sw_pct"], out["BAL_3_abs"], up_amp_abs=2.02, down_amp_abs=1.48,
        modulation_signal=mod3, modulation_strength=0.040, ref_balance=ref3
    )

    out["IKT_abs_plot"] = pct_to_absolute_weakly_modulated_amplitude(
        out["IKT_pct"], out["BAL_4_abs"], up_amp_abs=2.10, down_amp_abs=1.54,
        modulation_signal=mod4, modulation_strength=0.047, ref_balance=ref4
    )
    out["IHT_abs_plot"] = pct_to_absolute_weakly_modulated_amplitude(
        out["IHT_pct"], out["BAL_4_abs"], up_amp_abs=2.10, down_amp_abs=1.54,
        modulation_signal=mod4, modulation_strength=0.047, ref_balance=ref4
    )

    # Диагностика: фактическая амплитуда 1-го графика по циклам должна быть почти постоянной.
    # Для удобства сохраняем абсолютные отклонения от баланса.
    out["GDP_gap_abs"] = out["GDP_abs"] - out["BAL_1_abs"]
    out["GNI_gap_abs"] = out["GNI_abs"] - out["BAL_1_abs"]
    out["AD_gap_abs"] = out["AD_abs"] - out["BAL_2_abs"]
    out["AS_gap_abs"] = out["AS_abs"] - out["BAL_2_abs"]

    return out

def local_smooth_crossings_absolute(df: pd.DataFrame, events):
    """Финальная микродоводка только возле пересечений с линией баланса.
    На спаде сглаживание сильнее и чуть шире, чем на подъеме: цель — убрать остаточную
    ломкость в точке входа в баланс после пика, но не изменить общую геометрию цикла.
    """
    out = df.copy()
    idx = out["t"].to_numpy(dtype=float)

    cols = [
        "GDP_abs", "GNI_abs",
        "AD_abs", "AS_abs",
        "Pw_abs_plot", "Sw_abs_plot",
        "IKT_abs_plot", "IHT_abs_plot",
    ]

    for col in cols:
        base = out[col].to_numpy(dtype=float)
        sm1 = smooth_series(base, passes=1)
        sm2 = smooth_series(base, passes=2)
        sm3 = smooth_series(base, passes=3)
        sm4 = smooth_series(base, passes=4)

        total_weight = np.zeros_like(base)
        blend_accum = np.zeros_like(base)

        for ev in events:
            up = float(ev["up_cross"])
            down = float(ev["down_cross"])
            peak = float(ev["peak"])

            # На подъеме — только легкая шлифовка, чтобы не округлить уже выпрямленный верхний сегмент.
            w_up = 0.30 * bell(idx, up, 3.5) + 0.12 * bell(idx, up + 0.8, 5.4)

            # На спаде — более широкая и асимметричная шлифовка: чуть сильнее перед пересечением,
            # и немного слабее сразу после него.
            w_down_core = 0.78 * bell(idx, down, 5.3)
            w_down_pre = 0.34 * bell(idx, down - 2.0, 6.3)
            w_down_post = 0.18 * bell(idx, down + 1.1, 5.0)
            w_down = w_down_core + w_down_pre + w_down_post

            # Узкое локальное выравнивание точно возле down_cross.
            lo = max(0, int(round(down - 6)))
            hi = min(len(base) - 1, int(round(down + 6)))
            local_line = base.copy()
            if hi > lo:
                xs = np.arange(lo, hi + 1, dtype=float)
                line = np.interp(xs, [lo, hi], [base[lo], base[hi]])
                local_line[lo:hi + 1] = line

            # Дополнительное выравнивание на отрезке peak -> down_cross,
            # чтобы вход в баланс на спаде был мягче и без остаточного внешнего выгиба.
            seg_lo = max(0, int(round(peak + 0.34 * (down - peak))))
            seg_hi = max(seg_lo + 1, min(len(base) - 1, int(round(down + 1))))
            descent_line = base.copy()
            if seg_hi > seg_lo:
                xs = np.arange(seg_lo, seg_hi + 1, dtype=float)
                line = np.interp(xs, [seg_lo, seg_hi], [base[seg_lo], base[seg_hi]])
                descent_line[seg_lo:seg_hi + 1] = line

            up_target = 0.74 * sm1 + 0.26 * sm2
            down_target = (
                0.20 * base +
                0.20 * sm1 +
                0.24 * sm2 +
                0.18 * sm3 +
                0.08 * sm4 +
                0.06 * local_line +
                0.04 * descent_line
            )

            blend_accum += w_up * (up_target - base)
            blend_accum += w_down * (down_target - base)
            total_weight += w_up + w_down

        total_weight = clip(total_weight, 0.0, 0.90)
        out[col] = base + total_weight * blend_accum
        out[col] = (1.0 - total_weight) * base + total_weight * out[col]

    return out


def set_panel_limits(ax, series_list):
    ymax = max(float(np.max(np.asarray(s, dtype=float))) for s in series_list)
    ax.set_ylim(0.0, ymax * 1.08)


def plot_four_panels_absolute(df: pd.DataFrame, events, out_path: str = OUT_PNG):
    ensure_parent_dir(out_path)
    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"hspace": 0.10})

    phase_handles = [
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.0, label="дно цикла"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.05, label="пересечение линии баланса"),
        Line2D([0], [0], color="gray", linestyle="-.", linewidth=1.15, label="пик цикла"),
        Line2D([0], [0], color="gray", linestyle=(0, (6, 5)), linewidth=1.4, label="линия баланса (эндогенный тренд)"),
    ]

    for ax in axes:
        draw_phase_lines(ax, events)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("усл. абс. ед.")

    axes[0].plot(df["t"], df["BAL_1_abs"], color="gray", linestyle=(0, (6, 5)), linewidth=1.4, alpha=0.9)
    axes[0].plot(df["t"], df["GDP_abs"], linewidth=2.0, label="ВВП")
    axes[0].plot(df["t"], df["GNI_abs"], linewidth=2.0, label="ВНД")
    axes[0].set_title("ВВП / ВНД | абсолютные величины")
    set_panel_limits(axes[0], [df["GDP_abs"], df["GNI_abs"], df["BAL_1_abs"]])
    h0, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h0 + phase_handles, loc="upper left", ncol=2)

    axes[1].plot(df["t"], df["BAL_2_abs"], color="gray", linestyle=(0, (6, 5)), linewidth=1.4, alpha=0.9)
    axes[1].plot(df["t"], df["AS_abs"], linewidth=2.0, label="предложение")
    axes[1].plot(df["t"], df["AD_abs"], linewidth=2.0, label="спрос")
    axes[1].set_title("Спрос / предложение | абсолютные величины")
    set_panel_limits(axes[1], [df["AS_abs"], df["AD_abs"], df["BAL_2_abs"]])
    axes[1].legend(loc="upper left")

    axes[2].plot(df["t"], df["BAL_3_abs"], color="gray", linestyle=(0, (6, 5)), linewidth=1.4, alpha=0.9)
    axes[2].plot(df["t"], df["Pw_abs_plot"], linewidth=2.0, label="изымаемая прибыль")
    axes[2].plot(df["t"], df["Sw_abs_plot"], linewidth=2.0, label="изымаемые сбережения")
    axes[2].set_title("Изымаемая прибыль / Изымаемые сбережения | абсолютные величины")
    set_panel_limits(axes[2], [df["Pw_abs_plot"], df["Sw_abs_plot"], df["BAL_3_abs"]])
    axes[2].legend(loc="upper left")

    axes[3].plot(df["t"], df["BAL_4_abs"], color="gray", linestyle=(0, (6, 5)), linewidth=1.4, alpha=0.9)
    axes[3].plot(df["t"], df["IKT_abs_plot"], linewidth=2.0, label="совокупные инвестиции в физический капитал")
    axes[3].plot(df["t"], df["IHT_abs_plot"], linewidth=2.0, label="совокупные инвестиции в человеческий капитал")
    axes[3].set_title("Совокупные инвестиции в физический капитал / человеческий капитал | абсолютные величины")
    set_panel_limits(axes[3], [df["IKT_abs_plot"], df["IHT_abs_plot"], df["BAL_4_abs"]])
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Количество итераций")

    fig.suptitle(
        "Эндогенная саморегулируемая матричная модель | версия 23 | финальная микродоводка пересечений с линией баланса (особенно на спаде) | немного более мягкий вход в баланс после пика | эндогенный тренд капитала | почти постоянная амплитуда с очень слабой модуляцией",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)



def maybe_display_saved_png(path: str):
    """Если код запущен внутри notebook-ядра (а не через !python как подпроцесс),
    пытается показать сохраненный PNG прямо под ячейкой.
    При запуске через shell-команду в Colab inline-показ обычно недоступен,
    поэтому в этом случае функция тихо ничего не делает.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        # В shell-подпроцессе inline backend недоступен; безопаснее просто попытаться вывести Image.
        from IPython.display import display, Image
        if os.path.exists(path):
            display(Image(filename=path))
    except Exception:
        pass


def main(show_inline: bool = True):
    cfg = AMERConfig()
    df, events, chosen = build_display_data(cfg)
    df = enrich_with_panel_endogenous_weakly_modulated_amplitude(df)
    df = local_smooth_crossings_absolute(df, events)
    plot_four_panels_absolute(df, events, OUT_PNG)
    ensure_parent_dir(OUT_CSV)
    df.to_csv(OUT_CSV, index=False)

    print("A matrix (products x resources):")
    print(cfg.A)
    print("\nB matrix (resources x products):")
    print(cfg.B)
    print("\nВыбранные эндогенные циклы (по trough-to-trough):")
    for i, cyc in enumerate(chosen, start=1):
        print(f"Цикл {i}: start={cyc[0]}, end={cyc[1]}")
    print("\nКритические точки на отображаемой оси:")
    for ev in events:
        print(ev)

    print("\nМаксимальная ошибка тождества P_net = Pc - Pw:", float(np.max(np.abs(df["P_net"] - df["P_net_check"]))))
    print("Максимальная ошибка тождества S_net = Sc - Sw:", float(np.max(np.abs(df["S_net"] - df["S_net_check"]))))
    print("Максимальная ошибка тождества I_K^T = I_K^N + I_K^A + R_K^N + R_K^A:", float(np.max(np.abs(df["I_K_T_abs"] - df["I_K_T_check"]))))
    print("Максимальная ошибка тождества I_H^T = I_H^N + I_H^A + R_H^N + R_H^A:", float(np.max(np.abs(df["I_H_T_abs"] - df["I_H_T_check"]))))
    print(f"\nPNG сохранен: {OUT_PNG}")
    print(f"CSV сохранен: {OUT_CSV}")
    print("\nЕсли вы запускали через !python в Colab, покажите PNG отдельной ячейкой:")
    print("from IPython.display import Image, display")
    print(f'display(Image(filename=r"{OUT_PNG}"))')
    if show_inline:
        maybe_display_saved_png(OUT_PNG)


if __name__ == "__main__":
    main()
