import streamlit as st
import numpy as np
import pandas as pd

from core.params import POP
from core.model import kernels_one_state, dth_from_dae

st.set_page_config(
    page_title="呼吸道颗粒物沉积计算软件",
    layout="wide"
)

st.title("呼吸道颗粒物沉积计算软件")
st.caption("单粒径沉积分数计算")

# -----------------------------
# 中文标签
# -----------------------------
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

# -----------------------------
# 侧边栏参数
# -----------------------------
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

wind_speed = st.sidebar.number_input(
    "风速 U（m/s）",
    min_value=0.0,
    value=1.0,
    step=0.1
)

dae_um = st.sidebar.number_input(
    "空气动力学直径 dae（μm）",
    min_value=0.001,
    value=1.0,
    step=0.1,
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

# -----------------------------
# 计算按钮
# -----------------------------
if st.sidebar.button("计算沉积分数"):
    try:
        base = POP[pop_key]["base"]
        beh = POP[pop_key]["behaviors"][behavior_key]

        dae = np.array([dae_um], dtype=float)
        dth = dth_from_dae(dae, rho_g=rho_g, chi=chi)

        nose_breath = True if breathing_mode == "鼻呼吸" else False

        K = kernels_one_state(
            dae=dae,
            dth=dth,
            base=base,
            beh=beh,
            nose_breath=nose_breath,
            U_user=wind_speed
        )

        dep_sum = K["sum"]

        result = {}
        total_dep = 0.0

        for region in ["ET1", "ET2", "BB", "bb", "AI"]:
            value = float(np.asarray(dep_sum[region]).reshape(-1)[0])
            result[region] = value
            total_dep += value

        result_df = pd.DataFrame({
            "区域": [REGION_LABELS_ZH[r] for r in result.keys()] + ["总沉积分数"],
            "沉积分数": list(result.values()) + [total_dep]
        })

        st.subheader("计算结果")
        st.write(f"所选人群：{POP_LABELS_ZH.get(pop_key, pop_key)}")
        st.write(f"活动状态：{STATE_LABELS_ZH.get(behavior_key, behavior_key)}")
        st.write(f"空气动力学直径 dae = {dae_um:.3f} μm")
        st.write(f"换算后的热力学直径 dth = {float(dth[0]):.3f} μm")

        st.dataframe(result_df, use_container_width=True)

        st.subheader("区域沉积分数柱状图")
        chart_df = result_df.iloc[:-1].copy()
        st.bar_chart(chart_df.set_index("区域"))

    except Exception as e:
        st.error(f"计算失败：{e}")

else:
    st.info("请在左侧设置参数后，点击“计算沉积分数”。")
