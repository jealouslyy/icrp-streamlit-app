import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.params import POP
from core.model import kernels_one_state, dth_from_dae

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
# 小工具函数
# =========================
def diag_from_K(K, reg: str) -> np.ndarray:
    """
    从 kernels_one_state 返回的 K['sum'][reg] 中提取对角线，
    与原桌面版 main.py 的处理思路一致。
    """
    arr2d = np.asarray(K["sum"][reg], float)
    if arr2d.ndim < 2 or arr2d.size == 0:
        return np.array([], dtype=float)

    n = min(arr2d.shape[0], arr2d.shape[1])
    if n <= 0:
        return np.array([], dtype=float)

    idx = np.arange(n)
    y = arr2d[idx, idx].astype(float)

    # 若底层返回的是百分比而不是分数，则转成 0~1
    if np.nanmax(y) > 1.05:
        y = y / 100.0

    return y


def calc_single_dep(pop_key, behavior_key, nose_breath, wind_speed, dae_um, rho_g, chi):
    """
    单粒径沉积分数
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
    沉积分数曲线
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
    for region in REGIONS:
        y = diag_from_K(K, region)
        curve[region] = y

    n = min(len(curve[r]) for r in REGIONS)
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


# =========================
# 页面
# =========================
st.title("呼吸道颗粒物沉积计算软件")
st.subheader("第一页：沉积分数计算")

# ---------- 侧边栏 ----------
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

run_btn = st.sidebar.button("计算沉积分数")

# ---------- 主区 ----------
if run_btn:
    try:
        single = calc_single_dep(
            pop_key=pop_key,
            behavior_key=behavior_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            dae_um=dae_um,
            rho_g=rho_g,
            chi=chi
        )

        curve = calc_dep_curve(
            pop_key=pop_key,
            behavior_key=behavior_key,
            nose_breath=nose_breath,
            wind_speed=wind_speed,
            rho_g=rho_g,
            chi=chi
        )

        st.write(f"所选人群：{POP_LABELS_ZH.get(pop_key, pop_key)}")
        st.write(f"活动状态：{STATE_LABELS_ZH.get(behavior_key, behavior_key)}")
        st.write(f"空气动力学直径 dae = {dae_um:.3f} μm")
        st.write(f"换算后的热力学直径 dth = {float(single['dth'][0]):.3f} μm")

        # =====================
        # 1. 单粒径结果表
        # =====================
        result_df = pd.DataFrame({
            "区域": [REGION_LABELS_ZH[r] for r in REGIONS] + ["总沉积分数"],
            "沉积分数": [single["by_region"][r] for r in REGIONS] + [single["total"]]
        })

        st.subheader("单粒径沉积分数结果")
        st.dataframe(result_df, use_container_width=True)

        # =====================
        # 2. 柱状图
        # =====================
        st.subheader("区域沉积分数柱状图")
        chart_df = pd.DataFrame({
            "区域": [REGION_LABELS_ZH[r] for r in REGIONS],
            "沉积分数": [single["by_region"][r] for r in REGIONS]
        }).set_index("区域")
        st.bar_chart(chart_df)

        # =====================
        # 3. 曲线图
        # =====================
        st.subheader("分区沉积分数曲线")

        fig, ax = plt.subplots(figsize=(10, 5.5))

        color_map = {
            "ET1": "C0",
            "ET2": "C1",
            "BB": "C2",
            "bb": "C3",
            "AI": "C4",
            "Total": "#444444",
        }

        dae_curve = curve["dae"]

        for region in REGIONS:
            ax.plot(
                dae_curve,
                curve["curve"][region],
                label=REGION_LABELS_ZH[region],
                color=color_map[region],
                linewidth=2.0
            )

        ax.plot(
            dae_curve,
            curve["total"],
            label="总和",
            color=color_map["Total"],
            linewidth=2.8
        )

        ax.set_xscale("log")
        ax.set_xlim(dae_curve.min(), dae_curve.max())
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("颗粒空气动力学直径 (μm)")
        ax.set_ylabel("沉积分数")
        ax.grid(which="major", linestyle="--", alpha=0.35)
        ax.grid(which="minor", axis="x", linestyle=":", alpha=0.25)
        ax.legend(frameon=False, ncol=3)

        title_breath = "鼻呼吸" if nose_breath else "口呼吸"
        ax.set_title(
            f"分区沉积分数曲线：{POP_LABELS_ZH.get(pop_key, pop_key)} / "
            f"{STATE_LABELS_ZH.get(behavior_key, behavior_key)} / "
            f"{title_breath} / U={wind_speed:g}"
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"计算失败：{e}")

else:
    st.info("请在左侧设置参数后，点击“计算沉积分数”。")
