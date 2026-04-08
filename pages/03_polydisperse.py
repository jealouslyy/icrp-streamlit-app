import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.params import POP
from core.model import kernels_one_state, dth_from_dae

# =========================
# Matplotlib 设置
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

# =========================
# 常量
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
def diag_from_K(K, reg):
    arr2d = np.asarray(K["sum"][reg], dtype=float)
    n = min(arr2d.shape[0], arr2d.shape[1])
    y = arr2d[np.arange(n), np.arange(n)].astype(float)
    if np.nanmax(y) > 1.05:
        y = y / 100.0
    return np.clip(y, 0.0, 1.0)

def get_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return None

def get_ventilation_rate_m3_h(pop_key, behavior_key):
    beh = POP[pop_key]["behaviors"][behavior_key]

    direct = get_attr(beh, ["VE", "ve", "Vdot", "vdot", "Q", "q", "B", "b"])
    if direct is not None:
        return float(direct)

    vt = get_attr(beh, ["Vt", "vt", "tidal_volume"])
    fr = get_attr(beh, ["Fr", "fr", "f", "freq", "breathing_frequency"])

    if vt is not None and fr is not None:
        vt = float(vt)
        fr = float(fr)
        if vt > 10:
            vt_m3 = vt * 1e-6
        elif vt > 1:
            vt_m3 = vt * 1e-3
        else:
            vt_m3 = vt
        return vt_m3 * fr * 60.0

    raise ValueError("无法识别人群通气量参数。")

def convert_concentration_to_ug_m3(conc, unit):
    conc = np.asarray(conc, dtype=float)
    if unit == "μg/m³":
        return conc
    elif unit == "ng/m³":
        return conc / 1000.0
    elif unit == "mg/m³":
        return conc * 1000.0
    else:
        raise ValueError("不支持的浓度单位。")

def lognormal_mass_pdf(dp, mmad, gsd):
    dp = np.asarray(dp, dtype=float)
    sigma = np.log(gsd)
    mu = np.log(mmad)
    return (1 / (dp * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(dp) - mu) ** 2) / (2 * sigma ** 2)
    )

def build_polydisperse_distribution_multi(modes_df, total_conc_ug_m3, dp_min, dp_max, n_bins):
    edges = np.logspace(np.log10(dp_min), np.log10(dp_max), n_bins + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])
    widths_log = np.diff(np.log10(edges))

    peak_weights_list = []

    for _, row in modes_df.iterrows():
        mmad = float(row["mmad"])
        gsd = float(row["gsd"])
        frac = float(row["fraction"])

        pdf_i = lognormal_mass_pdf(mids, mmad, gsd)
        w_i = frac * pdf_i * widths_log
        peak_weights_list.append(w_i)

    if len(peak_weights_list) == 0:
        raise ValueError("请至少输入一个有效峰。")

    peak_weights_arr = np.vstack(peak_weights_list)   # (n_peak, n_bins)
    total_weights = np.sum(peak_weights_arr, axis=0)
    total_sum = np.sum(total_weights)

    if total_sum <= 0:
        raise ValueError("多峰分布权重计算失败，请检查 mmad、gsd 和 fraction 输入。")

    peak_weights_arr = peak_weights_arr / total_sum
    total_weights = np.sum(peak_weights_arr, axis=0)

    data = {
        "dp_min_um": edges[:-1],
        "dp_max_um": edges[1:],
        "dae_um": mids,
        "mass_fraction": total_weights,
        "conc_ug_m3": total_conc_ug_m3 * total_weights,
    }

    for j in range(peak_weights_arr.shape[0]):
        data[f"peak{j+1}_mass_fraction"] = peak_weights_arr[j]
        data[f"peak{j+1}_conc_ug_m3"] = total_conc_ug_m3 * peak_weights_arr[j]

    return pd.DataFrame(data)

def calc_dep(pop_key, behavior_key, nose_breath, wind_speed, dae_um, rho_g, chi):
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

    by_region = {r: diag_from_K(K, r) for r in REGIONS}
    total = np.clip(sum(by_region[r] for r in REGIONS), 0.0, 1.0)

    return {"dae": dae, "dth": dth, "by_region": by_region, "total": total}

def calc_polydisperse_weighted(
    pop_key,
    nose_breath,
    wind_speed,
    rho_g,
    chi,
    modes_df,
    total_conc,
    concentration_unit,
    dp_min,
    dp_max,
    n_bins,
    time_dict,
):
    total_conc_ug_m3 = float(convert_concentration_to_ug_m3([total_conc], concentration_unit)[0])

    dist_df = build_polydisperse_distribution_multi(
        modes_df=modes_df,
        total_conc_ug_m3=total_conc_ug_m3,
        dp_min=dp_min,
        dp_max=dp_max,
        n_bins=n_bins,
    )

    dae = dist_df["dae_um"].to_numpy()
    total_conc_arr = dist_df["conc_ug_m3"].to_numpy()

    total_result_df = dist_df.copy()
    by_state_rows = []

    # 初始化总剂量列
    total_result_df["dth_um"] = np.nan
    total_result_df["ET1_df"] = 0.0
    total_result_df["ET2_df"] = 0.0
    total_result_df["BB_df"] = 0.0
    total_result_df["bb_df"] = 0.0
    total_result_df["AI_df"] = 0.0
    total_result_df["Total_df"] = 0.0

    for r in REGIONS:
        total_result_df[f"{r}_dose_ug"] = 0.0

    n_peaks = len(modes_df)
    for i in range(n_peaks):
        for r in REGIONS:
            total_result_df[f"peak{i+1}_{r}_dose_ug"] = 0.0

    for behavior_key in BEHAVIOR_ORDER:
        t = float(time_dict.get(behavior_key, 0.0))
        if t <= 0:
            continue

        vent = get_ventilation_rate_m3_h(pop_key, behavior_key)
        dep = calc_dep(
            pop_key=pop_key,
            behavior_key=behavior_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            dae_um=dae,
            rho_g=rho_g,
            chi=chi,
        )

        inhaled_each_total = total_conc_arr * vent * t

        # 保存最后一个状态下的 dth
        total_result_df["dth_um"] = dep["dth"]

        # 对各状态的沉积分数做时间加权平均显示
        total_result_df["ET1_df"] += dep["by_region"]["ET1"] * t
        total_result_df["ET2_df"] += dep["by_region"]["ET2"] * t
        total_result_df["BB_df"] += dep["by_region"]["BB"] * t
        total_result_df["bb_df"] += dep["by_region"]["bb"] * t
        total_result_df["AI_df"] += dep["by_region"]["AI"] * t
        total_result_df["Total_df"] += dep["total"] * t

        for r in REGIONS:
            total_result_df[f"{r}_dose_ug"] += inhaled_each_total * dep["by_region"][r]

        # 每个峰分别计算各区域剂量
        for i in range(n_peaks):
            peak_conc = total_result_df[f"peak{i+1}_conc_ug_m3"].to_numpy()
            inhaled_each_peak = peak_conc * vent * t
            for r in REGIONS:
                total_result_df[f"peak{i+1}_{r}_dose_ug"] += inhaled_each_peak * dep["by_region"][r]

        state_total = sum(np.sum(inhaled_each_total * dep["by_region"][r]) for r in REGIONS)

        by_state_rows.append({
            "活动状态": STATE_LABELS_ZH[behavior_key],
            "暴露时长 (h)": t,
            "通气量 (m³/h)": vent,
            "吸入质量 (μg)": float(np.sum(inhaled_each_total)),
            "总沉积剂量 (μg)": float(state_total),
        })

    total_time = sum(float(time_dict.get(k, 0.0)) for k in BEHAVIOR_ORDER)
    if total_time <= 0:
        raise ValueError("四种活动状态的暴露时长均为 0，无法计算。")

    total_result_df["ET1_df"] /= total_time
    total_result_df["ET2_df"] /= total_time
    total_result_df["BB_df"] /= total_time
    total_result_df["bb_df"] /= total_time
    total_result_df["AI_df"] /= total_time
    total_result_df["Total_df"] /= total_time

    summary = {
        "ET1_total_ug": float(np.sum(total_result_df["ET1_dose_ug"])),
        "ET2_total_ug": float(np.sum(total_result_df["ET2_dose_ug"])),
        "BB_total_ug": float(np.sum(total_result_df["BB_dose_ug"])),
        "bb_total_ug": float(np.sum(total_result_df["bb_dose_ug"])),
        "AI_total_ug": float(np.sum(total_result_df["AI_dose_ug"])),
        "total_inhaled_ug": float(sum(x["吸入质量 (μg)"] for x in by_state_rows)),
        "by_state_df": pd.DataFrame(by_state_rows),
    }
    summary["Total_deposited_ug"] = (
        summary["ET1_total_ug"] + summary["ET2_total_ug"] +
        summary["BB_total_ug"] + summary["bb_total_ug"] + summary["AI_total_ug"]
    )

    return total_result_df, summary

def make_summary_df(summary):
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

def make_peak_contribution_df(result_df, modes_df):
    rows = []
    n_peaks = len(modes_df)

    for i in range(n_peaks):
        et1 = float(np.sum(result_df[f"peak{i+1}_ET1_dose_ug"]))
        et2 = float(np.sum(result_df[f"peak{i+1}_ET2_dose_ug"]))
        bb  = float(np.sum(result_df[f"peak{i+1}_BB_dose_ug"]))
        bb2 = float(np.sum(result_df[f"peak{i+1}_bb_dose_ug"]))
        ai  = float(np.sum(result_df[f"peak{i+1}_AI_dose_ug"]))

        rows.append({
            "峰": f"峰{i+1}",
            "鼻腔前部": et1,
            "鼻腔后部": et2,
            "支气管": bb,
            "细支气管": bb2,
            "肺泡区": ai,
            "总沉积剂量(μg)": et1 + et2 + bb + bb2 + ai
        })

    return pd.DataFrame(rows)

# =========================
# 绘图函数
# =========================
def plot_peak_distribution(result_df, modes_df):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    dae = result_df["dae_um"].to_numpy()

    for i in range(len(modes_df)):
        col = f"peak{i+1}_conc_ug_m3"
        if col in result_df.columns:
            ax.plot(
                dae,
                result_df[col],
                marker="o",
                linewidth=1.8,
                label=f"峰{i+1}"
            )

    ax.plot(
        dae,
        result_df["conc_ug_m3"],
        color="black",
        linewidth=2.8,
        label="总分布"
    )

    ax.set_xscale("log")
    ax.set_title("多峰分布中各峰对总质量浓度的贡献", fontsize=14, fontweight="bold")
    ax.set_xlabel("空气动力学直径 dae（μm）", fontsize=12, fontweight="bold")
    ax.set_ylabel("质量浓度（μg/m³）", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig

def plot_peak_contribution_bar(peak_df):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    bars = ax.bar(peak_df["峰"], peak_df["总沉积剂量(μg)"], width=0.62, alpha=0.9)

    ax.set_title("各峰对总沉积剂量的贡献", fontsize=14, fontweight="bold")
    ax.set_xlabel("分布峰", fontsize=12, fontweight="bold")
    ax.set_ylabel("总沉积剂量（μg）", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ymax = peak_df["总沉积剂量(μg)"].max() if len(peak_df) > 0 else 1
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    for bar, val in zip(bars, peak_df["总沉积剂量(μg)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ymax * 0.02 if ymax > 0 else 0.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    fig.tight_layout()
    return fig

def plot_region_bar(summary):
    labels = ["鼻腔前部", "鼻腔后部", "支气管", "细支气管", "肺泡区"]
    values = [
        summary["ET1_total_ug"],
        summary["ET2_total_ug"],
        summary["BB_total_ug"],
        summary["bb_total_ug"],
        summary["AI_total_ug"],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, values, width=0.62, alpha=0.9)

    ax.set_title("多分散气溶胶各区域汇总沉积剂量", fontsize=14, fontweight="bold")
    ax.set_xlabel("呼吸道区域", fontsize=12, fontweight="bold")
    ax.set_ylabel("沉积剂量（μg）", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ymax = max(values) if len(values) > 0 else 1
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ymax * 0.02 if ymax > 0 else 0.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    fig.tight_layout()
    return fig

# =========================
# 页面标题
# =========================
st.title("第三页：多分散气溶胶沉积计算")

with st.expander("当前页面功能说明", expanded=False):
    st.write(
        "本页基于一个或多个对数正态峰的叠加分布，输入各峰的中值粒径、几何标准差及质量分数，"
        "并结合总质量浓度与四种活动状态暴露时长，计算多分散气溶胶在呼吸道各区域的沉积剂量。"
    )

# =========================
# 侧边栏参数
# =========================
st.sidebar.header("参数设置")

pop_key = st.sidebar.selectbox(
    "选择人群",
    options=list(POP.keys()),
    format_func=lambda x: POP_LABELS_ZH.get(x, x)
)

breathing_mode = st.sidebar.selectbox("呼吸方式", ["鼻呼吸", "口呼吸"])
nose_breath = breathing_mode == "鼻呼吸"

wind_speed = st.sidebar.number_input("风速 U（m/s）", min_value=0.0, value=1.0, step=0.1)
rho_g = st.sidebar.number_input("颗粒密度 ρ（g/cm³）", min_value=0.1, value=1.5, step=0.1)
chi = st.sidebar.number_input("形状因子 χ", min_value=0.1, value=1.0, step=0.1)

st.sidebar.subheader("多峰分布")
total_conc = st.sidebar.number_input("总质量浓度", min_value=0.0, value=50.0, step=1.0)
concentration_unit = st.sidebar.selectbox("浓度单位", ["μg/m³", "ng/m³", "mg/m³"], index=0)

dp_min = st.sidebar.number_input("积分下限粒径（μm）", min_value=0.001, value=0.01, step=0.01, format="%.3f")
dp_max = st.sidebar.number_input("积分上限粒径（μm）", min_value=0.01, value=10.0, step=0.1)
n_bins = st.sidebar.slider("粒径划分数", min_value=10, max_value=100, value=40, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("四种活动状态暴露时长")
sleep_time_h = st.sidebar.number_input("睡眠时长（h）", min_value=0.0, value=8.0, step=0.5)
rest_time_h = st.sidebar.number_input("静坐时长（h）", min_value=0.0, value=8.0, step=0.5)
light_time_h = st.sidebar.number_input("轻度运动时长（h）", min_value=0.0, value=4.0, step=0.5)
heavy_time_h = st.sidebar.number_input("重度运动时长（h）", min_value=0.0, value=0.0, step=0.5)

default_modes = pd.DataFrame({
    "mmad": [0.08, 0.8],
    "gsd": [1.8, 2.2],
    "fraction": [0.4, 0.6],
})

st.markdown("### 多峰分布参数输入")
modes_df = st.data_editor(
    default_modes,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

time_dict = {
    "sleep": sleep_time_h,
    "rest": rest_time_h,
    "light": light_time_h,
    "heavy": heavy_time_h,
}
total_time_h = sum(time_dict.values())

# =========================
# 参数展示
# =========================
st.markdown("### 当前参数")
a1, a2, a3 = st.columns(3)
a1.write(f"**人群**：{POP_LABELS_ZH.get(pop_key, pop_key)}")
a2.write(f"**呼吸方式**：{breathing_mode}")
a3.write(f"**总暴露时长**：{total_time_h:.2f} h")

b1, b2, b3, b4 = st.columns(4)
b1.write(f"**分布峰数**：{len(modes_df)}")
b2.write(f"**总浓度**：{total_conc:.3f} {concentration_unit}")
b3.write(f"**粒径范围**：{dp_min:.3f}–{dp_max:.3f} μm")
b4.write(f"**粒径划分数**：{n_bins}")

st.dataframe(modes_df, use_container_width=True, hide_index=True)

if total_time_h > 24:
    st.warning("四种活动状态总时长超过 24 h，请检查输入。")

run_btn = st.button("计算多分散气溶胶沉积剂量", use_container_width=True)

if run_btn:
    try:
        modes_df = modes_df.copy()

        required_cols = ["mmad", "gsd", "fraction"]
        if not all(col in modes_df.columns for col in required_cols):
            raise ValueError("多峰分布表必须包含 mmad、gsd 和 fraction 三列。")

        for col in required_cols:
            modes_df[col] = pd.to_numeric(modes_df[col], errors="coerce")

        modes_df = modes_df.dropna(subset=required_cols)

        if len(modes_df) == 0:
            raise ValueError("请至少输入一个峰的参数。")

        if np.any(modes_df["mmad"] <= 0):
            raise ValueError("每个峰的 mmad 必须大于 0。")
        if np.any(modes_df["gsd"] <= 1):
            raise ValueError("每个峰的 gsd 必须大于 1。")
        if np.any(modes_df["fraction"] < 0):
            raise ValueError("每个峰的 fraction 不能为负值。")
        if dp_min <= 0 or dp_max <= 0 or dp_min >= dp_max:
            raise ValueError("粒径范围设置不正确，请确保下限 > 0 且上限 > 下限。")

        frac_sum = modes_df["fraction"].sum()
        if frac_sum <= 0:
            raise ValueError("fraction 总和必须大于 0。")

        modes_df["fraction"] = modes_df["fraction"] / frac_sum

        result_df, summary = calc_polydisperse_weighted(
            pop_key=pop_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            rho_g=rho_g,
            chi=chi,
            modes_df=modes_df,
            total_conc=total_conc,
            concentration_unit=concentration_unit,
            dp_min=dp_min,
            dp_max=dp_max,
            n_bins=n_bins,
            time_dict=time_dict,
        )

        peak_df = make_peak_contribution_df(result_df, modes_df)

        st.markdown("---")
        st.subheader("计算摘要")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("吸入总质量 (μg)", f"{summary['total_inhaled_ug']:.4f}")
        s2.metric("总沉积剂量 (μg)", f"{summary['Total_deposited_ug']:.4f}")
        s3.metric("粒径段数", f"{len(result_df)}")
        s4.metric("总暴露时长 (h)", f"{total_time_h:.2f}")

        st.subheader("各活动状态贡献")
        st.dataframe(summary["by_state_df"], use_container_width=True, hide_index=True)

        st.subheader("粒径分布与分段结果")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.subheader("各区域汇总沉积剂量")
        summary_df = make_summary_df(summary)
        show_df = summary_df.copy()
        show_df["沉积剂量 (μg)"] = show_df["沉积剂量 (μg)"].map(lambda x: f"{x:.6f}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.subheader("各峰对区域沉积的贡献")
        show_peak_df = peak_df.copy()
        for col in show_peak_df.columns[1:]:
            show_peak_df[col] = show_peak_df[col].map(lambda x: f"{x:.6f}")
        st.dataframe(show_peak_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_peak_distribution(result_df, modes_df), use_container_width=True)
        with c2:
            st.pyplot(plot_peak_contribution_bar(peak_df), use_container_width=True)

        st.subheader("区域总沉积剂量分布")
        st.pyplot(plot_region_bar(summary), use_container_width=True)

        csv1 = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载多分散分段结果 CSV",
            data=csv1,
            file_name="polydisperse_result.csv",
            mime="text/csv"
        )

        csv2 = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载区域汇总结果 CSV",
            data=csv2,
            file_name="polydisperse_summary.csv",
            mime="text/csv"
        )

        csv3 = peak_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载各峰贡献结果 CSV",
            data=csv3,
            file_name="polydisperse_peak_contribution.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"计算失败：{e}")
        st.exception(e)
