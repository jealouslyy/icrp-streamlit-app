import streamlit as st

st.set_page_config(page_title="呼吸道颗粒物沉积计算软件", layout="wide")

st.markdown("""
<style>
/* ===== 全局标题加粗一点 ===== */
h1, h2, h3, h4 {
    font-weight: 800 !important;
    color: #1f2a44 !important;
}

/* ===== 所有按钮基础样式 ===== */
div.stButton > button {
    font-weight: 700 !important;
    font-size: 20px !important;
    min-height: 56px !important;
    border-radius: 12px !important;
    border: 2px solid #c7d3e3 !important;
    background-color: #ffffff !important;
    color: #24364b !important;
    transition: all 0.2s ease-in-out !important;
}

/* ===== 普通按钮悬停 ===== */
div.stButton > button:hover {
    border-color: #6f8fb8 !important;
    color: #1d4f91 !important;
    background-color: #f7faff !important;
    box-shadow: 0 4px 12px rgba(80, 110, 150, 0.12) !important;
}

/* ===== 点击时轻微缩放 ===== */
div.stButton > button:active {
    transform: scale(0.985) !important;
}

/* ===== 聚焦效果 ===== */
div.stButton > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 0.18rem rgba(100, 140, 200, 0.16) !important;
}

/* ===== primary 按钮样式：主入口按钮 ===== */
div.stButton > button[kind="primary"] {
    background: #4f6fa3 !important;
    color: white !important;
    border: 2px solid #4f6fa3 !important;
    font-weight: 800 !important;
}

/* ===== primary 按钮悬停 ===== */
div.stButton > button[kind="primary"]:hover {
    background: #3f5f90 !important;
    border-color: #3f5f90 !important;
    color: white !important;
    box-shadow: 0 6px 14px rgba(63, 95, 144, 0.22) !important;
}

/* ===== primary 按钮点击 ===== */
div.stButton > button[kind="primary"]:active {
    background: #36527d !important;
    border-color: #36527d !important;
}

/* ===== expander 标题稍微加粗 ===== */
details summary {
    font-weight: 700 !important;
    color: #24364b !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 一些小工具
# =========================
def go_page(page_name: str):
    try:
        st.switch_page(page_name)
    except Exception:
        st.warning(f"当前环境未能自动跳转，请在左侧导航栏进入：{page_name}")

# =========================
# 页面主体
# =========================
st.title("呼吸道颗粒物沉积计算软件")
st.markdown("### 基于 ICRP 模型的颗粒物呼吸道沉积与剂量分析平台")

st.markdown(
    """
本平台面向颗粒物吸入暴露研究与结果展示，支持不同人群、呼吸方式和活动状态下的呼吸道沉积计算。  
可实现单粒径、粒径点列表、粒径段分布及多分散气溶胶条件下的沉积分数、沉积剂量与结果导出。
"""
)

# =========================
# 顶部快捷按钮
# =========================
b1, b2 = st.columns([1, 5])

with b1:
    if st.button("开始使用", type="primary", use_container_width=True):
        go_page("pages/01_呼吸道沉积分数计算.py")

st.markdown("---")

# =========================
# 功能入口区
# =========================
st.subheader("功能模块")

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("#### 沉积分数计算")
        st.markdown(
            """
支持单粒径、粒径点列表和粒径段输入，  
计算前鼻区域、后鼻-咽喉区域、支气管区域、细支气管区域与肺泡-间质区域的沉积分数。
"""
        )
        if st.button("进入沉积分数计算", key="go_frac", type="primary", use_container_width=True):
            go_page("pages/01_呼吸道沉积分数计算.py")

    with st.container(border=True):
        st.markdown("#### 多分散气溶胶计算")
        st.markdown(
            """
基于质量中值粒径和几何标准差，  
计算多分散气溶胶粒径分布及各区域沉积结果。
"""
        )
        if st.button("进入多分散气溶胶计算", key="go_poly", type="primary", use_container_width=True):
            go_page("pages/03_多分散气溶胶沉积计算.py")

with c2:
    with st.container(border=True):
        st.markdown("#### 沉积剂量计算")
        st.markdown(
            """
结合颗粒物浓度、暴露时长与活动状态，  
计算呼吸道各区域沉积剂量。
"""
        )
        if st.button("进入沉积剂量计算", key="go_dose", type="primary", use_container_width=True):
            go_page("pages/02_呼吸道沉积剂量计算.py")

    with st.container(border=True):
        st.markdown("#### 结果展示与导出")
        st.markdown(
            """
支持结果表格、图形展示及结果导出，  
便于后续整理、比较与分析。
"""
        )
        st.button("查看导出说明", key="export_note", use_container_width=True)

st.markdown("---")

# =========================
# 使用流程
# =========================
st.subheader("使用流程")

s1, s2, s3, s4 = st.columns(4)

with s1:
    with st.container(border=True):
        st.markdown("**步骤 1**")
        st.markdown("选择功能模块")
        st.caption("进入沉积分数、沉积剂量或多分散气溶胶计算页面")

with s2:
    with st.container(border=True):
        st.markdown("**步骤 2**")
        st.markdown("设置参数")
        st.caption("选择人群、呼吸方式、活动状态，并输入粒径或浓度参数")

with s3:
    with st.container(border=True):
        st.markdown("**步骤 3**")
        st.markdown("执行计算")
        st.caption("查看呼吸道各区域沉积分数、沉积剂量及图表结果")

with s4:
    with st.container(border=True):
        st.markdown("**步骤 4**")
        st.markdown("导出结果")
        st.caption("按需导出结果表格与图形，用于后续分析与展示")

st.markdown("---")

# =========================
# 模型说明
# =========================
with st.expander("模型依据与适用说明"):
    st.markdown(
        """
- 本平台基于 ICRP 呼吸道沉积模型构建。  
- 支持不同人群、呼吸方式与活动状态下的颗粒物沉积分析。  
- 可用于科研分析、教学展示及案例比较。  
- 结果解释应结合输入参数范围、粒径定义与具体应用场景综合判断。  
"""
    )

# =========================
# 页面底部说明
# =========================
st.caption("建议在情景对比分析中保持除目标变量外的其他参数一致，以便比较不同粒径或暴露条件对沉积结果的影响。")
