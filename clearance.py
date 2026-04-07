from matplotlib.dates import SA
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal
from scipy.integrate import odeint, cumulative_trapezoid as cumtrapz

from .params import(
    Material, AbsorptionParams,
    CLEARANCE_ABSORB_PROFILES,
    CLEARANCE_DEFAULTS,
    ClearanceTransferParams as TransferParams,
    ClearancePartitionParams as PartitionParams,
)

REGIONS = ("ET1", "ET2", "BB", "bb", "AI")

@dataclass
class HInputs:
    H_ET1: float
    H_ET2: float
    H_BB: float
    H_bb: float
    H_AI: float

def material_to_absorption(mat: Material) -> AbsorptionParams:
    return CLEARANCE_ABSORB_PROFILES[mat]

def compute_fs(dm_um: float, density: float, chi: float) -> float:
    threshold = 2.5 * np.sqrt(density / chi)
    return 0.5 if dm_um <= threshold else 0.5 * np.exp(-0.63 * (dm_um * np.sqrt(chi / density) - 2.5))

def build_partitions(fs: float, P: PartitionParams) -> Dict[str, float]:
    parts = {
        "P1": P.P1, "P2": P.P2, "P3": P.P3,
        "P4": max(P.P4_base - fs, 0.0),
        "P5": fs,
        "P6": P.P6,
        "P7": max(P.P7_base - fs, 0.0),
        "P8": fs,
        "P9": P.P9,
        "P11": P.P11,
        "P12": P.P12,
    }
    return parts

def solve_retention(
    H: HInputs,
    mat: Material,
    t_end_days: float,
    n_points: int,
    dm_um: float,
    density: float,
    chi: float,
    transfers: TransferParams = None,
    partitions: PartitionParams = None,
    ua: dict[str, float] | None = None,
    expo_hours: float | None = None,
    # ★ 新增两个参数：累计/文献风格需要的日程门控
    daily_on_hours: float | None = None,   # 每天暴露多少小时；None=不按日重复
    gate_mode: str = "ramp",               # "ramp"|"box"|"const"
    ramp_minutes: float = 60.0,            # ramp 平滑宽度(分钟)，仅用于 "ramp"
):
    """
    改进版：
    - GI 通量 = m11_15*R11 + m14_16*R14（ET2 与 ET1 两条支路）
    - 源项门控：支持一次性(箱形/平滑)、连续(const)、按日重复(daily_on_hours)
    - 暴露窗附近自适应加密时间网格
    - 新增：累计曲线（cum_*）与 Respiratory retention (ret_RT)
    - H 以 μg/h 输入，内部统一为 μg/day；Amount 单位 μg
    """
    ua = ua or {}
    if transfers is None:
        transfers = CLEARANCE_DEFAULTS["transfer"]
    if partitions is None:
        partitions = CLEARANCE_DEFAULTS["partition"]

    # ---- 速率不确定性修正（保持你原逻辑）----
    if "L_ALV_INT" in ua: transfers.m3_10 = ua["L_ALV_INT"]
    if "L_ALV_bb" in ua:  transfers.m6_10 = ua["L_ALV_bb"]
    if "L_INT_LNTH" in ua: transfers.m9_10 = ua["L_INT_LNTH"]
    Kpt = ua.get("Kpt", 1.0)
    for k in ["m1_4","m2_4","m3_4","m4_7","m5_7","m7_11","m8_11","m11_15","m12_13","m14_16"]:
        if hasattr(transfers, k):
            setattr(transfers, k, getattr(transfers, k) * Kpt)

    abs_p = material_to_absorption(mat)
    C_t = abs_p.fr * abs_p.sr
    C_b = (1.0 - abs_p.fr) * abs_p.ss

    fs = compute_fs(dm_um=dm_um, density=density, chi=chi)
    Ps = build_partitions(fs, partitions)

    # ---------- 自适应时间网格 ----------
    t_exp = (expo_hours or 0.0) / 24.0  # 天（仅用于一次性门控）
    dt_base = max(t_end_days / max(n_points, 50), 1e-9)
    ramp = max(5*dt_base, max(ramp_minutes/1440.0, 10/1440.0))  # ≥10min，且 ≥5*dt
    t_dense_end = min(t_end_days, (t_exp + 3*ramp) if daily_on_hours is None else min(t_end_days, 3.0))
    dt_dense = min(dt_base/4, ramp/12)
    n_dense = max(int(np.ceil(t_dense_end / max(dt_dense, 1e-9))), 50)
    n_coarse = max(int(np.ceil((t_end_days - t_dense_end) / max(dt_base, 1e-9))), 1)
    t1 = np.linspace(0.0, t_dense_end, n_dense, endpoint=True)
    t2 = np.linspace(t_dense_end, t_end_days, n_coarse+1, endpoint=True)[1:] if t_dense_end < t_end_days else np.array([])
    t = np.unique(np.concatenate([t1, t2]))
    R0 = 0.0

    # ---------- 源项门控 ----------
    def _gate_once(tt: float) -> float:
        if gate_mode == "const":
            return 1.0
        if gate_mode == "box":
            return 1.0 if tt <= t_exp else 0.0
        # ramp
        k = 4.0 / max(ramp, 1e-9)
        s_on  = 1.0 / (1.0 + np.exp(-k * (tt - 0.0)))
        s_off = 1.0 / (1.0 + np.exp(-k * (tt - t_exp)))
        return float(np.clip(s_on - s_off, 0.0, 1.0))

    def _gate_daily(tt: float, on_hours: float) -> float:
        if on_hours <= 0: return 0.0
        day = np.floor(tt)
        t_day = tt - day
        t_on, t_off = 0.0, on_hours / 24.0
        k = 4.0 / max(ramp, 1e-9)
        s_on  = 1.0 / (1.0 + np.exp(-k * (t_day - t_on)))
        s_off = 1.0 / (1.0 + np.exp(-k * (t_day - t_off)))
        return float(np.clip(s_on - s_off, 0.0, 1.0))

    def G(tt: float) -> float:
        if daily_on_hours is not None:
            return _gate_daily(tt, float(daily_on_hours))
        return _gate_once(tt)

    # μg/h -> μg/day
    H_day = np.array([H.H_ET1, H.H_ET2, H.H_BB, H.H_bb, H.H_AI], dtype=float) * 24.0
    def S_ET1(tt): return H_day[0] * G(tt)
    def S_ET2(tt): return H_day[1] * G(tt)
    def S_BB(tt):  return H_day[2] * G(tt)
    def S_bb(tt):  return H_day[3] * G(tt)
    def S_AI(tt):  return H_day[4] * G(tt)

    # ---------- 各仓室 ODE（保持你的形式） ----------
    def dR1(R1, tt): return -(transfers.m1_4 + C_t + C_b) * R1 + S_AI(tt) * Ps["P1"]
    def dR2(R2, tt): return -(transfers.m2_4 + C_t + C_b) * R2 + S_AI(tt) * Ps["P2"]
    def dR3(R3, tt): return -(transfers.m3_4 + C_t + C_b) * R3 + S_AI(tt) * Ps["P3"]
    R1 = odeint(lambda R, tt: dR1(R, tt), R0, t)[:, 0]
    R2 = odeint(lambda R, tt: dR2(R, tt), R0, t)[:, 0]
    R3 = odeint(lambda R, tt: dR3(R, tt), R0, t)[:, 0]

    def dR4(R4, tt):
        return -(transfers.m4_7 + C_t + C_b) * R4 + S_bb(tt) * Ps["P4"] \
               + transfers.m1_4 * np.interp(tt, t, R1) \
               + transfers.m2_4 * np.interp(tt, t, R2) \
               + transfers.m3_4 * np.interp(tt, t, R3)
    R4 = odeint(lambda R, tt: dR4(R, tt), R0, t)[:, 0]

    def dR5(R5, tt): return -(transfers.m5_7 + C_t + C_b) * R5 + S_bb(tt) * Ps["P5"]
    R5 = odeint(lambda R, tt: dR5(R, tt), R0, t)[:, 0]

    def dR6(R6, tt): return -(transfers.m6_10 + C_t + C_b) * R6 + S_bb(tt) * Ps["P6"]
    R6 = odeint(lambda R, tt: dR6(R, tt), R0, t)[:, 0]

    def dR7(R7, tt):
        return -(transfers.m7_11 + C_t + C_b) * R7 + S_BB(tt) * Ps["P7"] \
               + transfers.m4_7 * np.interp(tt, t, R4) \
               + transfers.m5_7 * np.interp(tt, t, R5)
    R7 = odeint(lambda R, tt: dR7(R, tt), R0, t)[:, 0]

    def dR8(R8, tt): return -(transfers.m8_11 + C_t + C_b) * R8 + S_BB(tt) * Ps["P8"]
    R8 = odeint(lambda R, tt: dR8(R, tt), R0, t)[:, 0]

    def dR9(R9, tt): return -(transfers.m9_10 + C_t + C_b) * R9 + S_BB(tt) * Ps["P9"]
    R9 = odeint(lambda R, tt: dR9(R, tt), R0, t)[:, 0]

    def dR10(R10, tt):
        return -(C_t + C_b) * R10 \
               + transfers.m3_10 * np.interp(tt, t, R3) \
               + transfers.m6_10 * np.interp(tt, t, R6) \
               + transfers.m9_10 * np.interp(tt, t, R9)
    R10 = odeint(lambda R, tt: dR10(R, tt), R0, t)[:, 0]

    def dR11(R11, tt):
        return -(transfers.m11_15 + C_t + C_b) * R11 + S_ET2(tt) * Ps["P11"] \
               + transfers.m7_11 * np.interp(tt, t, R7) \
               + transfers.m8_11 * np.interp(tt, t, R8)
    R11 = odeint(lambda R, tt: dR11(R, tt), R0, t)[:, 0]

    def dR12(R12, tt): return -(transfers.m12_13 + C_t + C_b) * R12 + S_ET2(tt) * Ps["P12"]
    R12 = odeint(lambda R, tt: dR12(R, tt), R0, t)[:, 0]

    def dR13(R13, tt): return -(C_t + C_b) * R13 + transfers.m12_13 * np.interp(tt, t, R12)
    R13 = odeint(lambda R, tt: dR13(R, tt), R0, t)[:, 0]

    def dR14(R14, tt): return -(transfers.m14_16) * R14 + S_ET1(tt)
    R14 = odeint(lambda R, tt: dR14(R, tt), R0, t)[:, 0]

    # ---------- GI 与 Blood ----------
    gi_flux   = transfers.m11_15 * R11 + transfers.m14_16 * R14  # μg/day
    blood_rate = (C_t + C_b) * (R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10 + R11 + R12 + R13)
    blood_cum = cumtrapz(blood_rate, t, initial=0.0)

    # ---------- 区域汇总 ----------
    R_AI  = R1 + R2 + R3 + R10
    R_bb  = R4 + R5 + R6
    R_BB  = R7 + R8 + R9
    R_ET2 = R11 + R12 + R13
    R_ET1 = R14

    # ---------- 各类“累计量”（文献图风格） ----------
    # 源项速率（进入各区的瞬时剂量率）：
    S_ET1_vec = np.array([S_ET1(tt) for tt in t])
    S_ET2_vec = np.array([S_ET2(tt) for tt in t])
    S_BB_vec  = np.array([S_BB (tt) for tt in t])
    S_bb_vec  = np.array([S_bb (tt) for tt in t])
    S_AI_vec  = np.array([S_AI (tt) for tt in t])
    src_total = S_ET1_vec + S_ET2_vec + S_BB_vec + S_bb_vec + S_AI_vec

    # ① 累计沉积到呼吸道（RT）
    cum_dep_RT = cumtrapz(src_total, t, initial=0.0)
    # ② 四区累计或五区累计（按需用）
    cum_ET1 = cumtrapz(S_ET1_vec, t, initial=0.0)
    cum_ET2 = cumtrapz(S_ET2_vec, t, initial=0.0)
    cum_BB  = cumtrapz(S_BB_vec , t, initial=0.0)
    cum_bb  = cumtrapz(S_bb_vec , t, initial=0.0)
    cum_AI  = cumtrapz(S_AI_vec , t, initial=0.0)
    cum_dep_4regions = cum_ET1 + cum_ET2 + cum_BB + cum_bb  # 若要 5 区总和可改为 + cum_AI

    # ③ GI 累计
    gi_cum = cumtrapz(gi_flux, t, initial=0.0)
    # ④ Blood 累计已是 blood_cum
    # ⑤ 呼吸道“保留量”（瞬时量）
    ret_RT = R_ET1 + R_ET2 + R_BB + R_bb + R_AI

    # ---------- 作图用平滑（不影响导出/积分） ----------
    def _smooth_for_plot(y, k=9):
        if k <= 1: return y
        kern = np.ones(k) / k
        return np.convolve(y, kern, mode='same')
    GI_plot = _smooth_for_plot(gi_flux, k=9)

    return t, {
        # 瞬时保留量（你现有的）
        "AI": R_AI, "bb": R_bb, "BB": R_BB, "ET2": R_ET2, "ET1": R_ET1,
        "Blood": blood_cum,
        "GI": gi_flux, "GI_plot": GI_plot,

        # 新增：累计与汇总
        "GI_cum": gi_cum,
        "cum_dep_RT": cum_dep_RT,
        "cum_dep_4regions": cum_dep_4regions,
        "cum_ET1": cum_ET1, "cum_ET2": cum_ET2, "cum_BB": cum_BB, "cum_bb": cum_bb, "cum_AI": cum_AI,
        "ret_RT": ret_RT,
    }

def run_clearance_with_rates(dep_result, ua=None, **kwargs):
    """
    简单封装：把沉积结果和UA参数传给 solve_retention。
    dep_result: dict，各区沉积分数
    ua: dict，不确定性参数（速率修正/Kpt）
    kwargs: 传给 solve_retention 的其它参数
    """
    return solve_retention(ua=ua, **kwargs)