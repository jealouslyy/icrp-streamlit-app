import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.params import POP
from core.model import kernels_one_state, dth_from_dae

# =========================
# Matplotlib 字体设置
# =========================
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10.5

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="Dose Calculation",
    layout="wide"
)

# =========================
# 基础字典
# =========================
REGIONS = ("ET1", "ET2", "BB", "bb", "AI")
BEHAVIOR_ORDER = ("sleep", "rest", "light", "heavy")

POP_LABELS_ZH = {
    "male_30y": "男性（30岁）",
    "male_15y": "男性（15岁）",
    "female_30y": "女性（30岁）",
    "female_15y": "女性（15岁）",
    "child_10y": "儿童（10岁）",
    "child_5y": "儿童（5岁）",
}

STATE_LABELS_ZH = {
    "sleep": "睡眠",
    "rest": "静坐",
    "light": "轻度运动",
    "heavy": "重度运动",
}

REGION_LABELS_ZH = {
    "ET1": "鼻腔前部",
    "ET2": "鼻腔后部",
    "BB": "支气管",
    "bb": "细支气管",
    "AI": "肺泡区",
}

# =========================
# 工具函数
# =========================
def diag_from_K(K, reg: str) -> np.ndarray:
    arr2d = np.asarray(K["sum"][reg], dtype=float)

    if arr2d.ndim < 2 or arr2d.size == 0:
        return np.array([], dtype=float)

    n = min(arr2d.shape[0], arr2d.shape[1])
    if n <= 0:
        return np.array([], dtype=float)

    idx = np.arange(n)
    y = arr2d[idx, idx].astype(float)

    if np.nanmax(y) > 1.05:
        y = y / 100.0

    return y


def get_behavior_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def get_ventilation_rate_m3_h(pop_key, behavior_key):
    """
    从 BehaviorParams 对象中提取通气量，统一换算为 m3/h
    """
    beh = POP[pop_key]["behaviors"][behavior_key]

    # 优先读取可能直接表示通气量的属性
    direct_val = get_behavior_attr(
        beh,
        ["VE", "ve", "Vdot", "vdot", "Q", "q", "B", "b"]
    )
    if direct_val is not None:
        direct_val = float(direct_val)
        if direct_val > 0:
            return direct_val

    # 用潮气量和呼吸频率计算
    vt = get_behavior_attr(beh, ["Vt", "vt", "tidal_volume"])
    fr = get_behavior_attr(beh, ["Fr", "fr", "f", "freq", "breathing_frequency"])

    if vt is not None and fr is not None:
        vt = float(vt)
        fr = float(fr)

        if vt > 10:
            vt_m3 = vt * 1e-6   # mL/次 -> m3/次
        elif vt > 1:
            vt_m3 = vt * 1e-3   # L/次 -> m3/次
        else:
            vt_m3 = vt          # 已经是 m3/次

        return vt_m3 * fr * 60.0

    raise ValueError(
        f"未能从 BehaviorParams 中识别通气量。可用属性："
        f"{vars(beh) if hasattr(beh, '__dict__') else dir(beh)}"
    )


def convert_concentration_to_ug_m3(conc, unit):
    conc = np.asarray(conc, dtype=float)

    if unit == "μg/m³":
        return conc
    elif unit == "mg/m³":
        return conc * 1000.0
    elif unit == "ng/m³":
        return conc / 1000.0
    else:
        raise ValueError("不支持的浓度单位。")

def bin_mid_geometric(dp_min, dp_max):
    """
    计算粒径段的几何均值粒径，作为该粒径段的代表粒径
    """
    dp_min = np.asarray(dp_min, dtype=float)
    dp_max = np.asarray(dp_max, dtype=float)

    if np.any(dp_min <= 0) or np.any(dp_max <= 0):
        raise ValueError("粒径段上下限必须大于 0。")
    if np.any(dp_max <= dp_min):
        raise ValueError("粒径段上限必须大于下限。")

    return np.sqrt(dp_min * dp_max)

def calc_dep_for_points(pop_key, behavior_key, nose_breath, wind_speed, dae_um, rho_g, chi):
    base = POP[pop_key]["base"]
    beh = POP[pop_key]["behaviors"][behavior_key]

    dae = np.asarray(dae_um, dtype=float)
    dth = dth_from_dae(dae, rho_g=rho_g, chi=chi)

    K = kernels_one_state(
        dae=dae,
        dth=dth,
        base=base,
        beh=beh,
        nose_breath=nose_breath,
        U_user=wind_speed
    )

    by_region = {}
    valid_lengths = []
    for region in REGIONS:
        y = diag_from_K(K, region)
        by_region[region] = y
        valid_lengths.append(len(y))

    n = min(valid_lengths) if valid_lengths else 0
    if n == 0:
        raise ValueError("未能获得有效的沉积分数计算结果。")

    dae = dae[:n]
    dth = dth[:n]

    for region in REGIONS:
        by_region[region] = np.clip(by_region[region][:n], 0.0, 1.0)

    total = np.clip(
        by_region["ET1"] + by_region["ET2"] + by_region["BB"] + by_region["bb"] + by_region["AI"],
        0.0, 1.0
    )

    return {
        "dae": dae,
        "dth": dth,
        "by_region": by_region,
        "total": total,
    }


def calc_dose_single_weighted(
    pop_key,
    nose_breath,
    wind_speed,
    dae_um,
    rho_g,
    chi,
    concentration,
    concentration_unit,
    time_dict,
):
    conc_ug_m3 = float(convert_concentration_to_ug_m3([concentration], concentration_unit)[0])

    total_inhaled_mass_ug = 0.0
    total_deposited_ug = 0.0
    by_region_total = {r: 0.0 for r in REGIONS}
    by_state_rows = []
    dth_ref = None

    for behavior_key in BEHAVIOR_ORDER:
        exposure_time_h = float(time_dict.get(behavior_key, 0.0))
        if exposure_time_h <= 0:
            continue

        dep = calc_dep_for_points(
            pop_key=pop_key,
            behavior_key=behavior_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            dae_um=[dae_um],
            rho_g=rho_g,
            chi=chi,
        )

        if dth_ref is None:
            dth_ref = float(dep["dth"][0])

        ventilation_m3_h = get_ventilation_rate_m3_h(pop_key, behavior_key)
        inhaled_mass_ug = conc_ug_m3 * ventilation_m3_h * exposure_time_h

        state_total_dep_ug = 0.0
        for region in REGIONS:
            region_dose = inhaled_mass_ug * float(dep["by_region"][region][0])
            by_region_total[region] += region_dose
            state_total_dep_ug += region_dose

        total_inhaled_mass_ug += inhaled_mass_ug
        total_deposited_ug += state_total_dep_ug

        by_state_rows.append({
            "活动状态": STATE_LABELS_ZH.get(behavior_key, behavior_key),
            "暴露时长 (h)": exposure_time_h,
            "通气量 (m³/h)": ventilation_m3_h,
            "吸入质量 (μg)": inhaled_mass_ug,
            "总沉积剂量 (μg)": state_total_dep_ug,
        })

    if len(by_state_rows) == 0:
        raise ValueError("四种活动状态的暴露时长均为 0，无法计算。")

    return {
        "dth_um": dth_ref,
        "conc_ug_m3": conc_ug_m3,
        "total_inhaled_mass_ug": total_inhaled_mass_ug,
        "total_deposited_ug": total_deposited_ug,
        "by_region_total_ug": by_region_total,
        "by_state_df": pd.DataFrame(by_state_rows),
    }


def calc_dose_points_weighted(
    pop_key,
    nose_breath,
    wind_speed,
    rho_g,
    chi,
    dae_um_list,
    conc_list,
    concentration_unit,
    time_dict,
):
    dae_um_arr = np.asarray(dae_um_list, dtype=float)
    conc_arr = np.asarray(conc_list, dtype=float)

    if len(dae_um_arr) == 0:
        raise ValueError("输入表不能为空。")
    if len(dae_um_arr) != len(conc_arr):
        raise ValueError("粒径列表与浓度列表长度不一致。")
    if np.any(dae_um_arr <= 0):
        raise ValueError("粒径必须大于 0。")
    if np.any(conc_arr < 0):
        raise ValueError("浓度不能为负值。")

    conc_ug_m3 = convert_concentration_to_ug_m3(conc_arr, concentration_unit)

    total_result_df = None
    by_state_rows = []

    for behavior_key in BEHAVIOR_ORDER:
        exposure_time_h = float(time_dict.get(behavior_key, 0.0))
        if exposure_time_h <= 0:
            continue

        ventilation_m3_h = get_ventilation_rate_m3_h(pop_key, behavior_key)

        dep = calc_dep_for_points(
            pop_key=pop_key,
            behavior_key=behavior_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            dae_um=dae_um_arr,
            rho_g=rho_g,
            chi=chi,
        )

        n = len(dep["dae"])
        conc_use = conc_ug_m3[:n]
        inhaled_mass_each_ug = conc_use * ventilation_m3_h * exposure_time_h

        state_df = pd.DataFrame({
            "dae_um": dep["dae"],
            "dth_um": dep["dth"],
            "conc_ug_m3": conc_use,
            "ET1_df": dep["by_region"]["ET1"],
            "ET2_df": dep["by_region"]["ET2"],
            "BB_df": dep["by_region"]["BB"],
            "bb_df": dep["by_region"]["bb"],
            "AI_df": dep["by_region"]["AI"],
            "Total_df": dep["total"],
            "ET1_dose_ug": inhaled_mass_each_ug * dep["by_region"]["ET1"],
            "ET2_dose_ug": inhaled_mass_each_ug * dep["by_region"]["ET2"],
            "BB_dose_ug": inhaled_mass_each_ug * dep["by_region"]["BB"],
            "bb_dose_ug": inhaled_mass_each_ug * dep["by_region"]["bb"],
            "AI_dose_ug": inhaled_mass_each_ug * dep["by_region"]["AI"],
        })

        if total_result_df is None:
            total_result_df = state_df.copy()
        else:
            for col in ["ET1_dose_ug", "ET2_dose_ug", "BB_dose_ug", "bb_dose_ug", "AI_dose_ug"]:
                total_result_df[col] += state_df[col]

        state_total_dep = float(
            np.sum(state_df["ET1_dose_ug"]) +
            np.sum(state_df["ET2_dose_ug"]) +
            np.sum(state_df["BB_dose_ug"]) +
            np.sum(state_df["bb_dose_ug"]) +
            np.sum(state_df["AI_dose_ug"])
        )

        by_state_rows.append({
            "活动状态": STATE_LABELS_ZH.get(behavior_key, behavior_key),
            "暴露时长 (h)": exposure_time_h,
            "通气量 (m³/h)": ventilation_m3_h,
            "吸入质量 (μg)": float(np.sum(inhaled_mass_each_ug)),
            "总沉积剂量 (μg)": state_total_dep,
        })

    if total_result_df is None:
        raise ValueError("四种活动状态的暴露时长均为 0，无法计算。")

    summary = {
        "ET1_total_ug": float(np.sum(total_result_df["ET1_dose_ug"])),
        "ET2_total_ug": float(np.sum(total_result_df["ET2_dose_ug"])),
        "BB_total_ug": float(np.sum(total_result_df["BB_dose_ug"])),
        "bb_total_ug": float(np.sum(total_result_df["bb_dose_ug"])),
        "AI_total_ug": float(np.sum(total_result_df["AI_dose_ug"])),
        "total_inhaled_ug": float(sum(row["吸入质量 (μg)"] for row in by_state_rows)),
        "by_state_df": pd.DataFrame(by_state_rows),
    }
    summary["Total_deposited_ug"] = (
        summary["ET1_total_ug"] +
        summary["ET2_total_ug"] +
        summary["BB_total_ug"] +
        summary["bb_total_ug"] +
        summary["AI_total_ug"]
    )

    return total_result_df, summary

def calc_dose_bins_weighted(
    pop_key,
    nose_breath,
    wind_speed,
    rho_g,
    chi,
    dp_min_list,
    dp_max_list,
    conc_list,
    concentration_unit,
    time_dict,
):
    dp_min_arr = np.asarray(dp_min_list, dtype=float)
    dp_max_arr = np.asarray(dp_max_list, dtype=float)
    conc_arr = np.asarray(conc_list, dtype=float)

    if len(dp_min_arr) == 0:
        raise ValueError("输入表不能为空。")
    if not (len(dp_min_arr) == len(dp_max_arr) == len(conc_arr)):
        raise ValueError("dp_min、dp_max 和浓度列表长度必须一致。")
    if np.any(dp_min_arr <= 0) or np.any(dp_max_arr <= 0):
        raise ValueError("粒径段上下限必须大于 0。")
    if np.any(dp_max_arr <= dp_min_arr):
        raise ValueError("粒径段上限必须大于下限。")
    if np.any(conc_arr < 0):
        raise ValueError("浓度不能为负值。")

    dae_mid_arr = bin_mid_geometric(dp_min_arr, dp_max_arr)
    conc_ug_m3 = convert_concentration_to_ug_m3(conc_arr, concentration_unit)

    total_result_df = None
    by_state_rows = []

    for behavior_key in BEHAVIOR_ORDER:
        exposure_time_h = float(time_dict.get(behavior_key, 0.0))
        if exposure_time_h <= 0:
            continue

        ventilation_m3_h = get_ventilation_rate_m3_h(pop_key, behavior_key)

        dep = calc_dep_for_points(
            pop_key=pop_key,
            behavior_key=behavior_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            dae_um=dae_mid_arr,
            rho_g=rho_g,
            chi=chi,
        )

        n = len(dep["dae"])
        dp_min_use = dp_min_arr[:n]
        dp_max_use = dp_max_arr[:n]
        conc_use = conc_ug_m3[:n]

        inhaled_mass_each_ug = conc_use * ventilation_m3_h * exposure_time_h

        state_df = pd.DataFrame({
            "dp_min_um": dp_min_use,
            "dp_max_um": dp_max_use,
            "dae_mid_um": dep["dae"],
            "dth_um": dep["dth"],
            "conc_ug_m3": conc_use,
            "ET1_df": dep["by_region"]["ET1"],
            "ET2_df": dep["by_region"]["ET2"],
            "BB_df": dep["by_region"]["BB"],
            "bb_df": dep["by_region"]["bb"],
            "AI_df": dep["by_region"]["AI"],
            "Total_df": dep["total"],
            "ET1_dose_ug": inhaled_mass_each_ug * dep["by_region"]["ET1"],
            "ET2_dose_ug": inhaled_mass_each_ug * dep["by_region"]["ET2"],
            "BB_dose_ug": inhaled_mass_each_ug * dep["by_region"]["BB"],
            "bb_dose_ug": inhaled_mass_each_ug * dep["by_region"]["bb"],
            "AI_dose_ug": inhaled_mass_each_ug * dep["by_region"]["AI"],
        })

        if total_result_df is None:
            total_result_df = state_df.copy()
        else:
            for col in ["ET1_dose_ug", "ET2_dose_ug", "BB_dose_ug", "bb_dose_ug", "AI_dose_ug"]:
                total_result_df[col] += state_df[col]

        state_total_dep = float(
            np.sum(state_df["ET1_dose_ug"]) +
            np.sum(state_df["ET2_dose_ug"]) +
            np.sum(state_df["BB_dose_ug"]) +
            np.sum(state_df["bb_dose_ug"]) +
            np.sum(state_df["AI_dose_ug"])
        )

        by_state_rows.append({
            "活动状态": STATE_LABELS_ZH.get(behavior_key, behavior_key),
            "暴露时长 (h)": exposure_time_h,
            "通气量 (m³/h)": ventilation_m3_h,
            "吸入质量 (μg)": float(np.sum(inhaled_mass_each_ug)),
            "总沉积剂量 (μg)": state_total_dep,
        })

    if total_result_df is None:
        raise ValueError("四种活动状态的暴露时长均为 0，无法计算。")

    summary = {
        "ET1_total_ug": float(np.sum(total_result_df["ET1_dose_ug"])),
        "ET2_total_ug": float(np.sum(total_result_df["ET2_dose_ug"])),
        "BB_total_ug": float(np.sum(total_result_df["BB_dose_ug"])),
        "bb_total_ug": float(np.sum(total_result_df["bb_dose_ug"])),
        "AI_total_ug": float(np.sum(total_result_df["AI_dose_ug"])),
        "total_inhaled_ug": float(sum(row["吸入质量 (μg)"] for row in by_state_rows)),
        "by_state_df": pd.DataFrame(by_state_rows),
    }
    summary["Total_deposited_ug"] = (
        summary["ET1_total_ug"] +
        summary["ET2_total_ug"] +
        summary["BB_total_ug"] +
        summary["bb_total_ug"] +
        summary["AI_total_ug"]
    )

    return total_result_df, summary

def make_single_result_df(weighted_result):
    rows = []
    for region in REGIONS:
        rows.append({
            "区域": REGION_LABELS_ZH[region],
            "区域沉积剂量 (μg)": weighted_result["by_region_total_ug"][region],
        })

    rows.append({
        "区域": "总沉积",
        "区域沉积剂量 (μg)": weighted_result["total_deposited_ug"],
    })
    return pd.DataFrame(rows)


def make_points_summary_df(summary):
    return pd.DataFrame({
        "区域": ["鼻腔前部", "鼻腔后部", "支气管", "细支气管", "肺泡区", "总沉积"],
        "沉积剂量 (μg)": [
            summary["ET1_total_ug"],
            summary["ET2_total_ug"],
            summary["BB_total_ug"],
            summary["bb_total_ug"],
            summary["AI_total_ug"],
            summary["Total_deposited_ug"],
        ]
    })


def plot_single_region_dose(weighted_result):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    labels = [REGION_LABELS_ZH[r] for r in REGIONS]
    values = [weighted_result["by_region_total_ug"][r] for r in REGIONS]

    bars = ax.bar(labels, values, width=0.62, alpha=0.9)

    ax.set_ylabel("区域沉积剂量（μg）", fontsize=13, fontweight="bold")
    ax.set_xlabel("呼吸道区域", fontsize=13, fontweight="bold")
    ax.set_title("呼吸道各区域沉积剂量分布", fontsize=15, fontweight="bold", pad=12)

    ax.tick_params(axis="x", labelsize=12, width=1.1, length=5)
    ax.tick_params(axis="y", labelsize=11, width=1.1, length=5)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ymax = max(values) if len(values) > 0 else 1
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02 if ymax > 0 else 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold"
        )

    fig.tight_layout()
    return fig


def plot_points_summary(summary):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    labels = ["鼻腔前部", "鼻腔后部", "支气管", "细支气管", "肺泡区"]
    values = [
        summary["ET1_total_ug"],
        summary["ET2_total_ug"],
        summary["BB_total_ug"],
        summary["bb_total_ug"],
        summary["AI_total_ug"],
    ]

    bars = ax.bar(labels, values, width=0.62, alpha=0.9)

    ax.set_ylabel("汇总沉积剂量（μg）", fontsize=13, fontweight="bold")
    ax.set_xlabel("呼吸道区域", fontsize=13, fontweight="bold")
    ax.set_title("各区域汇总沉积剂量", fontsize=15, fontweight="bold", pad=12)

    ax.tick_params(axis="x", labelsize=12, width=1.1, length=5)
    ax.tick_params(axis="y", labelsize=11, width=1.1, length=5)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ymax = max(values) if len(values) > 0 else 1
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02 if ymax > 0 else 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold"
        )

    fig.tight_layout()
    return fig


# =========================
# 页面标题
# =========================
st.title("第二页：沉积剂量计算")

with st.expander("当前页面功能说明", expanded=False):
    st.write("本页用于根据颗粒物浓度、四种活动状态暴露时长和呼吸参数计算呼吸道各区域的沉积剂量。")
    st.write("当前支持两种输入方式：单粒径输入、粒径点列表输入。")
    st.write("剂量结果采用睡眠、静坐、轻度运动和重度运动四种状态分别计算后加权求和。")

# =========================
# 侧边栏参数
# =========================
st.sidebar.header("参数设置")

input_mode = st.sidebar.radio(
    "输入方式",
    ["单粒径", "粒径点-浓度", "粒径段-浓度"]
)

pop_key = st.sidebar.selectbox(
    "选择人群",
    options=list(POP.keys()),
    format_func=lambda x: POP_LABELS_ZH.get(x, x)
)

breathing_mode = st.sidebar.selectbox(
    "呼吸方式",
    options=["鼻呼吸", "口呼吸"]
)
nose_breath = breathing_mode == "鼻呼吸"

wind_speed = st.sidebar.number_input(
    "风速 U（m/s）",
    min_value=0.0,
    value=1.0,
    step=0.1
)

rho_g = st.sidebar.number_input(
    "颗粒密度 ρ（g/cm³）",
    min_value=0.1,
    value=1.5,
    step=0.1
)

chi = st.sidebar.number_input(
    "形状因子 χ",
    min_value=0.1,
    value=1.0,
    step=0.1
)

concentration_unit = st.sidebar.selectbox(
    "浓度单位",
    options=["μg/m³", "ng/m³", "mg/m³"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("四种活动状态暴露时长")

sleep_time_h = st.sidebar.number_input("睡眠时长（h）", min_value=0.0, value=8.0, step=0.5)
rest_time_h = st.sidebar.number_input("静坐时长（h）", min_value=0.0, value=8.0, step=0.5)
light_time_h = st.sidebar.number_input("轻度运动时长（h）", min_value=0.0, value=4.0, step=0.5)
heavy_time_h = st.sidebar.number_input("重度运动时长（h）", min_value=0.0, value=0.0, step=0.5)

total_time_h = sleep_time_h + rest_time_h + light_time_h + heavy_time_h
st.sidebar.markdown(f"**总暴露时长：{total_time_h:.2f} h**")

time_dict = {
    "sleep": sleep_time_h,
    "rest": rest_time_h,
    "light": light_time_h,
    "heavy": heavy_time_h,
}

# =========================
# 当前参数展示
# =========================
st.markdown("### 当前参数")
c1, c2, c3 = st.columns(3)
c1.write(f"**输入方式**：{input_mode}")
c2.write(f"**人群**：{POP_LABELS_ZH.get(pop_key, pop_key)}")
c3.write("**活动状态**：四种活动状态下的时间加权")

c4, c5, c6, c7 = st.columns(4)
c4.write(f"**呼吸方式**：{'鼻呼吸' if nose_breath else '口呼吸'}")
c5.write(f"**风速**：{wind_speed:.2f} m/s")
c6.write(f"**密度**：{rho_g:.2f} g/cm³")
c7.write(f"**形状因子**：{chi:.2f}")

st.write(
    f"**浓度单位**：{concentration_unit} ｜ "
    f"**睡眠** {sleep_time_h:.1f} h ｜ "
    f"**静坐** {rest_time_h:.1f} h ｜ "
    f"**轻度运动** {light_time_h:.1f} h ｜ "
    f"**重度运动** {heavy_time_h:.1f} h"
)

if total_time_h > 24:
    st.warning("四种活动状态的总时长已超过 24 h，请检查输入。")

# =========================
# 模式 1：单粒径
# =========================
if input_mode == "单粒径":
    dae_um = st.number_input(
        "空气动力学直径 dae（μm）",
        min_value=0.001,
        value=0.1,
        step=0.01,
        format="%.3f"
    )

    concentration = st.number_input(
        "颗粒物浓度",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    run_single_btn = st.button("计算单粒径沉积剂量", use_container_width=True)

    if run_single_btn:
        try:
            weighted_result = calc_dose_single_weighted(
                pop_key=pop_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                dae_um=dae_um,
                rho_g=rho_g,
                chi=chi,
                concentration=concentration,
                concentration_unit=concentration_unit,
                time_dict=time_dict,
            )

            st.markdown("---")
            st.subheader("计算摘要")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("热力学直径 dth (μm)", f"{weighted_result['dth_um']:.3f}")
            s2.metric("吸入总质量 (μg)", f"{weighted_result['total_inhaled_mass_ug']:.4f}")
            s3.metric("总沉积剂量 (μg)", f"{weighted_result['total_deposited_ug']:.4f}")
            s4.metric("总暴露时长 (h)", f"{total_time_h:.2f}")

            st.subheader("各活动状态贡献")
            st.dataframe(weighted_result["by_state_df"], use_container_width=True, hide_index=True)

            st.subheader("区域沉积剂量结果表")
            result_df = make_single_result_df(weighted_result)
            show_df = result_df.copy()
            show_df["区域沉积剂量 (μg)"] = show_df["区域沉积剂量 (μg)"].map(lambda x: f"{x:.6f}")
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            st.subheader("区域沉积剂量柱状图")
            fig = plot_single_region_dose(weighted_result)
            st.pyplot(fig, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载单粒径剂量结果 CSV",
                data=csv_data,
                file_name="dose_single_weighted_result.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"计算失败：{e}")

# =========================
# 模式 2：粒径点列表
# =========================
elif input_mode == "粒径点-浓度":
    st.subheader("粒径点-浓度输入表")

    default_df = pd.DataFrame({
        "dae_um": [0.01, 0.03, 0.1, 0.3, 1.0],
        "conc": [10.0, 15.0, 25.0, 12.0, 5.0]
    })

    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config = {
            "dae_um": st.column_config.NumberColumn("粒径（μm）"),
            "conc": st.column_config.NumberColumn("浓度"),
        }
    )

    run_points_btn = st.button("计算粒径点列表沉积剂量", use_container_width=True)

    if run_points_btn:
        try:
            df_clean = edited_df.copy()

            if "dae_um" not in df_clean.columns or "conc" not in df_clean.columns:
                raise ValueError("输入表必须包含 dae_um 和 conc 两列。")

            df_clean = df_clean.dropna(subset=["dae_um", "conc"])
            if len(df_clean) == 0:
                raise ValueError("输入表为空，请填写粒径和浓度数据。")

            df_clean["dae_um"] = pd.to_numeric(df_clean["dae_um"], errors="coerce")
            df_clean["conc"] = pd.to_numeric(df_clean["conc"], errors="coerce")
            df_clean = df_clean.dropna(subset=["dae_um", "conc"])

            if len(df_clean) == 0:
                raise ValueError("输入表中的数据无法识别为数值。")

            result_df, summary = calc_dose_points_weighted(
                pop_key=pop_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                rho_g=rho_g,
                chi=chi,
                dae_um_list=df_clean["dae_um"].to_numpy(),
                conc_list=df_clean["conc"].to_numpy(),
                concentration_unit=concentration_unit,
                time_dict=time_dict,
            )

            st.markdown("---")
            st.subheader("计算摘要")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("吸入总质量 (μg)", f"{summary['total_inhaled_ug']:.4f}")
            s2.metric("总沉积剂量 (μg)", f"{summary['Total_deposited_ug']:.4f}")
            s3.metric("总暴露时长 (h)", f"{total_time_h:.2f}")
            s4.metric("粒径点数", f"{len(result_df)}")

            st.subheader("各活动状态贡献")
            st.dataframe(summary["by_state_df"], use_container_width=True, hide_index=True)

            st.subheader("各粒径点计算结果")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            st.subheader("各区域汇总沉积剂量")
            summary_df = make_points_summary_df(summary)
            show_summary_df = summary_df.copy()
            show_summary_df["沉积剂量 (μg)"] = show_summary_df["沉积剂量 (μg)"].map(lambda x: f"{x:.6f}")
            st.dataframe(show_summary_df, use_container_width=True, hide_index=True)

            st.subheader("区域汇总沉积剂量柱状图")
            fig = plot_points_summary(summary)
            st.pyplot(fig, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载粒径点列表结果 CSV",
                data=csv_data,
                file_name="dose_points_weighted_result.csv",
                mime="text/csv"
            )

            summary_csv_data = summary_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载区域汇总剂量 CSV",
                data=summary_csv_data,
                file_name="dose_points_weighted_summary.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"计算失败: {e}")
            
elif input_mode == "粒径段-浓度":
    st.subheader("粒径段-浓度输入表")

    default_df = pd.DataFrame({
        "dp_min_um": [0.01, 0.03, 0.1, 0.3, 1.0],
        "dp_max_um": [0.03, 0.1, 0.3, 1.0, 3.0],
        "conc": [10.0, 15.0, 25.0, 12.0, 5.0]
    })

    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
        "dp_min_um": st.column_config.NumberColumn("粒径下限（μm）"),
        "dp_max_um": st.column_config.NumberColumn("粒径上限（μm）"),
        "conc": st.column_config.NumberColumn("浓度"),
    )

    run_bins_btn = st.button("计算粒径段列表沉积剂量", use_container_width=True)

    if run_bins_btn:
        try:
            df_clean = edited_df.copy()

            required_cols = ["dp_min_um", "dp_max_um", "conc"]
            for col in required_cols:
                if col not in df_clean.columns:
                    raise ValueError(f"输入表必须包含 {col} 列。")

            df_clean = df_clean.dropna(subset=required_cols)
            if len(df_clean) == 0:
                raise ValueError("输入表为空，请填写粒径段和浓度数据。")

            for col in required_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            df_clean = df_clean.dropna(subset=required_cols)

            if len(df_clean) == 0:
                raise ValueError("输入表中的数据无法识别为数值。")

            result_df, summary = calc_dose_bins_weighted(
                pop_key=pop_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                rho_g=rho_g,
                chi=chi,
                dp_min_list=df_clean["dp_min_um"].to_numpy(),
                dp_max_list=df_clean["dp_max_um"].to_numpy(),
                conc_list=df_clean["conc"].to_numpy(),
                concentration_unit=concentration_unit,
                time_dict=time_dict,
            )

            st.markdown("---")
            st.subheader("计算摘要")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("吸入总质量 (μg)", f"{summary['total_inhaled_ug']:.4f}")
            s2.metric("总沉积剂量 (μg)", f"{summary['Total_deposited_ug']:.4f}")
            s3.metric("总暴露时长 (h)", f"{total_time_h:.2f}")
            s4.metric("粒径段数", f"{len(result_df)}")

            st.subheader("各活动状态贡献")
            st.dataframe(summary["by_state_df"], use_container_width=True, hide_index=True)

            st.subheader("各粒径段计算结果")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            st.subheader("各区域汇总沉积剂量")
            summary_df = make_points_summary_df(summary)
            show_summary_df = summary_df.copy()
            show_summary_df["沉积剂量 (μg)"] = show_summary_df["沉积剂量 (μg)"].map(lambda x: f"{x:.6f}")
            st.dataframe(show_summary_df, use_container_width=True, hide_index=True)

            st.subheader("区域汇总沉积剂量柱状图")
            fig = plot_points_summary(summary)
            st.pyplot(fig, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载粒径段列表结果 CSV",
                data=csv_data,
                file_name="dose_bins_weighted_result.csv",
                mime="text/csv"
            )

            summary_csv_data = summary_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载区域汇总剂量 CSV",
                data=summary_csv_data,
                file_name="dose_bins_weighted_summary.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"计算失败：{e}")
