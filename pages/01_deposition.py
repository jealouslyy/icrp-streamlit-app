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
# 统一绘图风格
# =========================
REGION_COLORS = {
    "ET1": "#4C78A8",   # 蓝
    "ET2": "#72B7B2",   # 青
    "BB":  "#54A24B",   # 绿
    "bb":  "#F58518",   # 橙
    "AI":  "#E45756",   # 红
}

TOTAL_COLOR = "#222222"
BAR_COLOR = "#6FA8DC"
BAR_EDGE = "#2F5597"

PEAK_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"
]

def apply_ax_style(ax):
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.30)
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.tick_params(axis="x", labelsize=11, width=1.0, length=4)
    ax.tick_params(axis="y", labelsize=11, width=1.0, length=4)

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="呼吸道颗粒物沉积计算软件",
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
    """
    从 kernels_one_state 返回的 K['sum'][reg] 中提取对角线，
    与桌面版 main.py 的处理思路保持一致。
    """
    arr2d = np.asarray(K["sum"][reg], dtype=float)

    if arr2d.ndim < 2 or arr2d.size == 0:
        return np.array([], dtype=float)

    n = min(arr2d.shape[0], arr2d.shape[1])
    if n <= 0:
        return np.array([], dtype=float)

    idx = np.arange(n)
    y = arr2d[idx, idx].astype(float)

    # 若底层返回的是百分比而不是分数，则统一转换为 0~1
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


def calc_dep_curve(pop_key, behavior_key, nose_breath, wind_speed, rho_g, chi):
    """
    计算沉积分数曲线
    """
    base = POP[pop_key]["base"]
    beh = POP[pop_key]["behaviors"][behavior_key]

    dae = np.logspace(-3, 1, 240)
    dth = dth_from_dae(dae, rho_g=rho_g, chi=chi)

    K = kernels_one_state(
        dae=dae,
        dth=dth,
        base=base,
        beh=beh,
        nose_breath=nose_breath,
        U_user=wind_speed
    )

    curve = {}
    valid_lengths = []

    for region in REGIONS:
        y = diag_from_K(K, region)
        curve[region] = y
        valid_lengths.append(len(y))

    n = min(valid_lengths) if valid_lengths else 0
    if n == 0:
        raise ValueError("未能获得有效的沉积分数曲线数据。")

    dae = dae[:n]

    for region in REGIONS:
        curve[region] = np.clip(curve[region][:n], 0.0, 1.0)

    total = np.clip(
        curve["ET1"] + curve["ET2"] + curve["BB"] + curve["bb"] + curve["AI"],
        0.0, 1.0
    )

    return {
        "dae": dae,
        "curve": curve,
        "total": total,
    }


def make_single_result_df(single_result):
    return pd.DataFrame({
        "区域": [REGION_LABELS_ZH[r] for r in REGIONS] + ["总沉积分数"],
        "沉积分数": [single_result["by_region"][r] for r in REGIONS] + [single_result["total"]]
    })


def plot_dep_curve(curve_result, pop_key, behavior_key, nose_breath, wind_speed):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    dae_curve = curve_result["dae"]

    for region in REGIONS:
        ax.plot(
            dae_curve,
            curve_result["curve"][region],
            label=REGION_LABELS_ZH[region],
            linewidth=2.4,
            color=REGION_COLORS[region],
            alpha=0.95
        )

    ax.plot(
        dae_curve,
        curve_result["total"],
        label="总沉积分数",
        linewidth=3.0,
        color=TOTAL_COLOR,
        linestyle="-"
    )

    ax.set_xscale("log")
    ax.set_xlim(dae_curve.min(), dae_curve.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("颗粒空气动力学直径（μm）", fontsize=13, fontweight="bold")
    ax.set_ylabel("沉积分数", fontsize=13, fontweight="bold")

    apply_ax_style(ax)
    ax.grid(which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.30)
    ax.grid(which="minor", axis="x", linestyle=":", linewidth=0.7, alpha=0.20)

    title_breath = "鼻呼吸" if nose_breath else "口呼吸"
    ax.set_title(
        f"分区沉积分数曲线：{POP_LABELS_ZH.get(pop_key, pop_key)} / "
        f"{STATE_LABELS_ZH.get(behavior_key, behavior_key)} / "
        f"{title_breath} / U={wind_speed:g} m/s",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=10.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.tight_layout()
    return fig


# =========================
# 页面标题
# =========================
st.title("呼吸道颗粒物沉积计算软件")
st.subheader("第一页：沉积分数计算")

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
run_single_btn = st.sidebar.button("计算单粒径沉积分数", use_container_width=True)
run_curve_btn = st.sidebar.button("绘制分区沉积分数曲线", use_container_width=True)
run_all_btn = st.sidebar.button("全部计算", use_container_width=True)

# =========================
# 页面说明
# =========================
with st.expander("当前页面功能说明", expanded=False):
    st.write("本页用于计算单粒径颗粒在呼吸道各区域的沉积分数，并绘制不同粒径下的分区沉积分数曲线。")

# =========================
# 计算逻辑
# =========================
do_single = run_single_btn or run_all_btn
do_curve = run_curve_btn or run_all_btn

if do_single or do_curve:
    try:
        st.markdown("### 当前参数")
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.write(f"**人群**：{POP_LABELS_ZH.get(pop_key, pop_key)}")
        col_info2.write(f"**活动状态**：{STATE_LABELS_ZH.get(behavior_key, behavior_key)}")
        col_info3.write(f"**呼吸方式**：{'鼻呼吸' if nose_breath else '口呼吸'}")

        col_info4, col_info5, col_info6, col_info7 = st.columns(4)
        col_info4.write(f"**风速**：{wind_speed:.2f} m/s")
        col_info5.write(f"**dae**：{dae_um:.3f} μm")
        col_info6.write(f"**密度**：{rho_g:.2f} g/cm³")
        col_info7.write(f"**形状因子**：{chi:.2f}")

        # ---------- 单粒径 ----------
        if do_single:
            single = calc_single_dep(
                pop_key=pop_key,
                behavior_key=behavior_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                dae_um=dae_um,
                rho_g=rho_g,
                chi=chi
            )

            st.markdown("---")
            st.subheader("单粒径沉积分数结果")

            st.write(f"换算后的热力学直径 dth = {float(single['dth'][0]):.3f} μm")

            result_df = make_single_result_df(single)
            show_df = result_df.copy()
            show_df["沉积分数"] = show_df["沉积分数"].map(lambda x: f"{x:.4f}")

            st.dataframe(show_df, use_container_width=True, hide_index=True)

            st.subheader("区域沉积分数柱状图")
            chart_df = pd.DataFrame({
                "区域": [REGION_LABELS_ZH[r] for r in REGIONS],
                "沉积分数": [single["by_region"][r] for r in REGIONS]
            }).set_index("区域")
            st.bar_chart(chart_df)

        # ---------- 曲线 ----------
        if do_curve:
            curve = calc_dep_curve(
                pop_key=pop_key,
                behavior_key=behavior_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed,
                rho_g=rho_g,
                chi=chi
            )

            st.markdown("---")
            st.subheader("分区沉积分数曲线")
            fig = plot_dep_curve(
                curve_result=curve,
                pop_key=pop_key,
                behavior_key=behavior_key,
                nose_breath=nose_breath,
                wind_speed=wind_speed
            )
            st.pyplot(fig, use_container_width=True)

            curve_df = pd.DataFrame({
                "dae_um": curve["dae"],
                "ET1": curve["curve"]["ET1"],
                "ET2": curve["curve"]["ET2"],
                "BB": curve["curve"]["BB"],
                "bb": curve["curve"]["bb"],
                "AI": curve["curve"]["AI"],
                "Total": curve["total"]
            })

            csv_data = curve_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载曲线数据 CSV",
                data=csv_data,
                file_name="deposition_curve.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"计算失败：{e}")

else:
    st.info("请在左侧设置参数后，点击“计算单粒径沉积分数”“绘制分区沉积分数曲线”或“全部计算”。")
