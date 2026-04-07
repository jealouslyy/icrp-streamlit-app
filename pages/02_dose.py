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

    # 1) 直接读取可能表示通气量的属性
    direct_val = get_behavior_attr(
        beh,
        ["VE", "ve", "Vdot", "vdot", "Q", "q", "B", "b"]
    )
    if direct_val is not None:
        direct_val = float(direct_val)
        if direct_val > 0:
            return direct_val

    # 2) 用潮气量和呼吸频率计算
    vt = get_behavior_attr(beh, ["Vt", "vt", "tidal_volume"])
    fr = get_behavior_attr(beh, ["Fr", "fr", "f", "freq", "breathing_frequency"])

    if vt is not None and fr is not None:
        vt = float(vt)
        fr = float(fr)

        # 自动判断 Vt 单位
        if vt > 10:
            # 常见情况：mL/次
            vt_m3 = vt * 1e-6
        elif vt > 1:
            # 常见情况：L/次
            vt_m3 = vt * 1e-3
        else:
            # 已经是 m3/次
            vt_m3 = vt

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


def calc_dep_for_points(pop_key, behavior_key, nose_breath, wind_speed, dae_um, rho_g, chi):
    """
    对一组粒径点计算各区域沉积分数
    """
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


def calc_dose_single(
    pop_key,
    behavior_key,
    nose_breath,
    wind_speed,
    dae_um,
    rho_g,
    chi,
    concentration,
    concentration_unit,
    exposure_time_h,
):
    dep = calc_dep_for_points(
        pop_key=pop_key,
        behavior_key=behavior_key,
        nose_breath=nose_breath,
        wind_speed=wind_speed,
        dae_um=[dae_um],
        rho_g=rho_g,
        chi=chi,
    )

    ventilation_m3_h = get_ventilation_rate_m3_h(pop_key, behavior_key)
    conc_ug_m3 = float(convert_concentration_to_ug_m3([concentration], concentration_unit)[0])

    inhaled_mass_ug = conc_ug_m3 * ventilation_m3_h * exposure_time_h

    by_region_dose = {}
    for region in REGIONS:
        by_region_dose[region] = inhaled_mass_ug * float(dep["by_region"][region][0])

    total_deposited_ug = inhaled_mass_ug * float(dep["total"][0])

    return {
        "dep": dep,
        "ventilation_m3_h": ventilation_m3_h,
        "conc_ug_m3": conc_ug_m3,
        "inhaled_mass_ug": inhaled_mass_ug,
        "by_region_dose_ug": by_region_dose,
        "total_deposited_ug": total_deposited_ug,
    }


def calc_dose_points(
    pop_key,
    behavior_key,
    nose_breath,
    wind_speed,
    rho_g,
    chi,
    dae_um_list,
    conc_list,
    concentration_unit,
    exposure_time_h,
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

    ventilation_m3_h = get_ventilation_rate_m3_h(pop_key, behavior_key)
    conc_ug_m3 = convert_concentration_to_ug_m3(conc_arr, concentration_unit)

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
    dae_um_arr = dae_um_arr[:n]
    conc_ug_m3 = conc_ug_m3[:n]

    inhaled_mass_each_ug = conc_ug_m3 * ventilation_m3_h * exposure_time_h

    by_region_dose = {}
    for region in REGIONS:
        by_region_dose[region] = inhaled_mass_each_ug * dep["by_region"][region]

    result_df = pd.DataFrame({
        "dae_um": dep["dae"],
        "dth_um": dep["dth"],
        "conc_ug_m3": conc_ug_m3,
        "ET1_df": dep["by_region"]["ET1"],
        "ET2_df": dep["by_region"]["ET2"],
        "BB_df": dep["by_region"]["BB"],
        "bb_df": dep["by_region"]["bb"],
        "AI_df": dep["by_region"]["AI"],
        "Total_df": dep["total"],
        "ET1_dose_ug": by_region_dose["ET1"],
        "ET2_dose_ug": by_region_dose["ET2"],
        "BB_dose_ug": by_region_dose["BB"],
        "bb_dose_ug": by_region_dose["bb"],
        "AI_dose_ug": by_region_dose["AI"],
    })

    summary = {
        "ventilation_m3_h": ventilation_m3_h,
        "total_inhaled_ug": float(np.sum(inhaled_mass_each_ug)),
        "ET1_total_ug": float(np.sum(by_region_dose["ET1"])),
        "ET2_total_ug": float(np.sum(by_region_dose["ET2"])),
        "BB_total_ug": float(np.sum(by_region_dose["BB"])),
        "bb_total_ug": float(np.sum(by_region_dose["bb"])),
        "AI_total_ug": float(np.sum(by_region_dose["AI"])),
    }
    summary["Total_deposited_ug"] = (
        summary["ET1_total_ug"]
        + summary["ET2_total_ug"]
        + summary["BB_total_ug"]
        + summary["bb_total_ug"]
        + summary["AI_total_ug"]
    )

    return result_df, summary


def make_single_result_df(dose_result):
    rows = []
    for region in REGIONS:
        rows.append({
            "区域": REGION_LABELS_ZH[region],
            "沉积分数": dose_result["dep"]["by_region"][region][0],
            "区域沉积剂量 (μg)": dose_result["by_region_dose_ug"][region],
        })

    rows.append({
        "区域": "总沉积",
        "沉积分数": dose_result["dep"]["total"][0],
        "区域沉积剂量 (μg)": dose_result["total_deposited_ug"],
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


def plot_single_region_dose(dose_result):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    labels = [REGION_LABELS_ZH[r] for r in REGIONS]
    values = [dose_result["by_region_dose_ug"][r] for r in REGIONS]

    ax.bar(labels, values)
    ax.set_ylabel("区域沉积剂量（μg）", fontsize=12)
    ax.set_xlabel("呼吸道区域", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("各呼吸道区域沉积剂量", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_points_summary(summary):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    labels = ["ET1", "ET2", "BB", "bb", "AI"]
    values = [
        summary["ET1_total_ug"],
        summary["ET2_total_ug"],
        summary["BB_total_ug"],
        summary["bb_total_ug"],
        summary["AI_total_ug"],
    ]

    ax.bar(labels, values)
    ax.set_ylabel("汇总沉积剂量（μg）", fontsize=12)
    ax.set_xlabel("呼吸道区域", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("各区域汇总沉积剂量", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


# =========================
# 页面标题
# =========================
st.title("第二页：沉积剂量计算")

with st.expander("当前页面功能说明", expanded=False):
    st.write("本页用于根据颗粒物浓度、暴露时长和呼吸参数计算呼吸道各区域的沉积剂量。")
    st.write("当前支持两种输入方式：单粒径输入、粒径点列表输入。")
    st.write("后续可继续扩展为粒径段输入与多分散沉积剂量计算。")

# =========================
# 侧边栏参数
# =========================
st.sidebar.header("参数设置")

input_mode = st.sidebar.radio(
    "输入方式",
    ["单粒径", "粒径点列表"]
)

pop_key = st.sidebar.selectbox(
    "选择人群",
    options=list(POP.keys()),
    format_func=lambda x: POP_LABELS_ZH.get(x, x)
)

behavior_key = st.sidebar.selectbox(
    "选择活动状态",
    options=list(POP[pop_key]["behaviors"].keys()),
    format_func=lambda x: STATE_LABELS_ZH.get(x, x)
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

exposure_time_h = st.sidebar.number_input(
    "暴露时长（h）",
    min_value=0.0,
    value=1.0,
    step=0.5
)

# =========================
# 当前参数展示
# =========================
st.markdown("### 当前参数")
c1, c2, c3 = st.columns(3)
c1.write(f"**输入方式**：{input_mode}")
c2.write(f"**人群**：{POP_LABELS_ZH.get(pop_key, pop_key)}")
c3.write(f"**活动状态**：{STATE_LABELS_ZH.get(behavior_key, behavior_key)}")

c4, c5, c6, c7 = st.columns(4)
c4.write(f"**呼吸方式**：{'鼻呼吸' if nose_breath else '口呼吸'}")
c5.write(f"**风速**：{wind_speed:.2f} m/s")
c6.write(f"**密度**：{rho_g:.2f} g/cm³")
c7.write(f"**形状因子**：{chi:.2f}")

st.write(f"**浓度单位**：{concentration_unit}；**暴露时长**：{exposure_time_h:.2f} h")

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
            dose_result = calc_dose_single(
                pop_key=pop_key,
                behavior_key=behavior_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                dae_um=dae_um,
                rho_g=rho_g,
                chi=chi,
                concentration=concentration,
                concentration_unit=concentration_unit,
                exposure_time_h=exposure_time_h,
            )

            st.markdown("---")
            st.subheader("计算摘要")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("热力学直径 dth (μm)", f"{float(dose_result['dep']['dth'][0]):.3f}")
            s2.metric("通气量 (m³/h)", f"{dose_result['ventilation_m3_h']:.4f}")
            s3.metric("吸入总质量 (μg)", f"{dose_result['inhaled_mass_ug']:.4f}")
            s4.metric("总沉积剂量 (μg)", f"{dose_result['total_deposited_ug']:.4f}")

            st.subheader("区域沉积剂量结果表")
            result_df = make_single_result_df(dose_result)
            show_df = result_df.copy()
            show_df["沉积分数"] = show_df["沉积分数"].map(lambda x: f"{x:.4f}")
            show_df["区域沉积剂量 (μg)"] = show_df["区域沉积剂量 (μg)"].map(lambda x: f"{x:.6f}")
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            st.subheader("区域沉积剂量柱状图")
            fig = plot_single_region_dose(dose_result)
            st.pyplot(fig, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载单粒径剂量结果 CSV",
                data=csv_data,
                file_name="dose_single_result.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"计算失败：{e}")

# =========================
# 模式 2：粒径点列表
# =========================
elif input_mode == "粒径点列表":
    st.subheader("粒径点-浓度输入表")

    default_df = pd.DataFrame({
        "dae_um": [0.01, 0.03, 0.1, 0.3, 1.0],
        "conc": [10.0, 15.0, 25.0, 12.0, 5.0]
    })

    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True
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

            result_df, summary = calc_dose_points(
                pop_key=pop_key,
                behavior_key=behavior_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                rho_g=rho_g,
                chi=chi,
                dae_um_list=df_clean["dae_um"].to_numpy(),
                conc_list=df_clean["conc"].to_numpy(),
                concentration_unit=concentration_unit,
                exposure_time_h=exposure_time_h,
            )

            st.markdown("---")
            st.subheader("计算摘要")

            s1, s2, s3 = st.columns(3)
            s1.metric("通气量 (m³/h)", f"{summary['ventilation_m3_h']:.4f}")
            s2.metric("吸入总质量 (μg)", f"{summary['total_inhaled_ug']:.4f}")
            s3.metric("总沉积剂量 (μg)", f"{summary['Total_deposited_ug']:.4f}")

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
                file_name="dose_points_result.csv",
                mime="text/csv"
            )

            summary_csv_data = summary_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载区域汇总剂量 CSV",
                data=summary_csv_data,
                file_name="dose_points_summary.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"计算失败：{e}")
