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


def calc_single_dep(pop_key, behavior_key, nose_breath, wind_speed, dae_um, rho_g, chi):
    """
    计算单粒径沉积分数
    """
    base = POP[pop_key]["base"]
    beh = POP[pop_key]["behaviors"][behavior_key]

    dae = np.array([dae_um], dtype=float)
    dth = dth_from_dae(dae, rho_g=rho_g, chi=chi)

    K = kernels_one_state(
        dae=dae,
        dth=dth,
        base=base,
        beh=beh,
        nose_breath=nose_breath,
        U_user=wind_speed
    )

    result = {}
    total_dep = 0.0

    for region in REGIONS:
        y = diag_from_K(K, region)
        value = float(y[0]) if len(y) > 0 else 0.0
        value = float(np.clip(value, 0.0, 1.0))
        result[region] = value
        total_dep += value

    total_dep = float(np.clip(total_dep, 0.0, 1.0))

    return {
        "dae": dae,
        "dth": dth,
        "by_region": result,
        "total": total_dep,
    }


def get_ventilation_rate_m3_h(pop_key, behavior_key):
    """
    尝试从参数中提取通气量，统一换算为 m3/h
    这里做了多种字段兼容，避免因为你的 core.params 结构略有不同而直接报错
    """
    beh = POP[pop_key]["behaviors"][behavior_key]

    # 常见直接给通气量的字段
    for key in ["VE", "ve", "Vdot", "vdot", "Q", "q", "B", "b"]:
        if key in beh:
            val = float(beh[key])
            if val > 0:
                return val

    # 若提供潮气量 Vt + 呼吸频率 Fr，则换算
    vt_keys = ["Vt", "vt", "tidal_volume"]
    fr_keys = ["Fr", "fr", "f", "freq", "breathing_frequency"]

    vt = None
    fr = None

    for key in vt_keys:
        if key in beh:
            vt = float(beh[key])
            break

    for key in fr_keys:
        if key in beh:
            fr = float(beh[key])
            break

    if vt is not None and fr is not None:
        # 如果 vt 很小，默认按 m3/次 处理；
        # 如果 vt 数值像 500 这种，则可能是 mL/次
        if vt > 10:
            vt_m3 = vt * 1e-6
        elif vt > 1:
            vt_m3 = vt * 1e-3
        else:
            vt_m3 = vt

        return vt_m3 * fr * 60.0

    raise ValueError("未能从 POP 参数中识别该活动状态下的通气量，请检查 core.params 中 behaviors 的字段名。")


def calc_dose(
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
    """
    基础剂量计算：
    吸入量 = 浓度 × 通气量 × 暴露时间
    区域剂量 = 吸入量 × 该区域沉积分数
    """

    dep = calc_single_dep(
        pop_key=pop_key,
        behavior_key=behavior_key,
        nose_breath=nose_breath,
        wind_speed=wind_speed,
        dae_um=dae_um,
        rho_g=rho_g,
        chi=chi
    )

    ventilation_m3_h = get_ventilation_rate_m3_h(pop_key, behavior_key)

    # 统一浓度单位到 μg/m3
    if concentration_unit == "μg/m³":
        conc_ug_m3 = concentration
    elif concentration_unit == "mg/m³":
        conc_ug_m3 = concentration * 1000.0
    elif concentration_unit == "ng/m³":
        conc_ug_m3 = concentration / 1000.0
    else:
        raise ValueError("不支持的浓度单位。")

    inhaled_mass_ug = conc_ug_m3 * ventilation_m3_h * exposure_time_h

    by_region_dose = {}
    for region in REGIONS:
        by_region_dose[region] = inhaled_mass_ug * dep["by_region"][region]

    total_deposited_ug = inhaled_mass_ug * dep["total"]

    return {
        "dep": dep,
        "ventilation_m3_h": ventilation_m3_h,
        "conc_ug_m3": conc_ug_m3,
        "inhaled_mass_ug": inhaled_mass_ug,
        "by_region_dose_ug": by_region_dose,
        "total_deposited_ug": total_deposited_ug,
    }


def make_dose_result_df(dose_result):
    rows = []

    for region in REGIONS:
        rows.append({
            "区域": REGION_LABELS_ZH[region],
            "沉积分数": dose_result["dep"]["by_region"][region],
            "区域沉积剂量 (μg)": dose_result["by_region_dose_ug"][region],
        })

    rows.append({
        "区域": "总沉积",
        "沉积分数": dose_result["dep"]["total"],
        "区域沉积剂量 (μg)": dose_result["total_deposited_ug"],
    })

    return pd.DataFrame(rows)


def plot_region_dose(dose_result):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

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


# =========================
# 页面标题
# =========================
st.title("呼吸道颗粒物沉积计算软件")
st.subheader("第二页：沉积剂量计算")

with st.expander("当前页面功能说明", expanded=False):
    st.write("本页用于基于单粒径颗粒浓度、暴露时间和人群呼吸参数，计算各呼吸道区域的沉积剂量。")
    st.write("当前为基础版：支持单粒径、单浓度输入。后续可继续扩展为粒径分布剂量和多分散剂量计算。")

# =========================
# 侧边栏参数
# =========================
st.sidebar.header("参数设置")

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

dae_um = st.sidebar.number_input(
    "空气动力学直径 dae（μm）",
    min_value=0.001,
    value=0.1,
    step=0.01,
    format="%.3f"
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

st.sidebar.markdown("---")
concentration = st.sidebar.number_input(
    "颗粒物浓度",
    min_value=0.0,
    value=50.0,
    step=1.0
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

run_dose_btn = st.sidebar.button("计算沉积剂量", use_container_width=True)

# =========================
# 计算逻辑
# =========================
if run_dose_btn:
    try:
        dose_result = calc_dose(
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

        st.markdown("### 当前参数")
        c1, c2, c3 = st.columns(3)
        c1.write(f"**人群**：{POP_LABELS_ZH.get(pop_key, pop_key)}")
        c2.write(f"**活动状态**：{STATE_LABELS_ZH.get(behavior_key, behavior_key)}")
        c3.write(f"**呼吸方式**：{'鼻呼吸' if nose_breath else '口呼吸'}")

        c4, c5, c6, c7 = st.columns(4)
        c4.write(f"**风速**：{wind_speed:.2f} m/s")
        c5.write(f"**dae**：{dae_um:.3f} μm")
        c6.write(f"**浓度**：{concentration:.3f} {concentration_unit}")
        c7.write(f"**暴露时长**：{exposure_time_h:.2f} h")

        st.markdown("---")
        st.subheader("计算摘要")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("热力学直径 dth (μm)", f"{float(dose_result['dep']['dth'][0]):.3f}")
        s2.metric("通气量 (m³/h)", f"{dose_result['ventilation_m3_h']:.4f}")
        s3.metric("吸入总质量 (μg)", f"{dose_result['inhaled_mass_ug']:.4f}")
        s4.metric("总沉积剂量 (μg)", f"{dose_result['total_deposited_ug']:.4f}")

        st.subheader("区域沉积剂量结果表")
        dose_df = make_dose_result_df(dose_result)
        show_df = dose_df.copy()
        show_df["沉积分数"] = show_df["沉积分数"].map(lambda x: f"{x:.4f}")
        show_df["区域沉积剂量 (μg)"] = show_df["区域沉积剂量 (μg)"].map(lambda x: f"{x:.6f}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.subheader("区域沉积剂量柱状图")
        fig = plot_region_dose(dose_result)
        st.pyplot(fig, use_container_width=True)

        export_df = dose_df.copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下载剂量结果 CSV",
            data=csv_data,
            file_name="dose_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"计算失败：{e}")

else:
    st.info("请在左侧设置参数后，点击“计算沉积剂量”。")
