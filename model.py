import numpy as np
from typing import Dict
from .params import BaseParams, BehaviorParams
# 滑移系数
def Cc(d):
    C = 1 + 0.0712 / d * (2.514 + 0.8 * np.exp(-0.55 * d / 0.0712))
    return C

# 空气动力学直径→热力学直径
def dth_from_dae(dae, rho_g=1.5, chi=1, tol=1e-9, maxit=50):
    dae = np.asarray(dae, dtype=float)
    Cae = Cc(dae)
    d = dae * np.sqrt(chi / rho_g)
    for _ in range(maxit):
        d_new = dae * np.sqrt((chi / rho_g) * Cae / Cc(d))
        if np.all(np.abs(d_new - d) / np.maximum(d_new, 1e-30) < tol):
            break
        d = d_new
    return d_new # μm

# 扩散系数 (cm2/s)
def CD(C, d):
    k_B = 1.3806E-23    # 玻尔兹曼常数
    T = 313             # 温度(K)

    D = k_B * T * C / (3 * np.pi * 0.000019 * d * 0.000001) * 10000
    return D

# 颗粒物可吸入性计算
def cal_in(d, U):
    η1 = 1 - 0.5 * (1 - (0.00076 * d ** 2.8 + 1) ** -1) + 0.00001 * U ** 2.75 * np.exp(0.055 * d)
    return η1

# ET1区域空气动力学吸入
def cal_ae_in_ET1(d, V, Fn, SF_t):
    ηAE = 0.5 * (1 - 1 / (0.0003 * (d ** 2 * V * Fn * SF_t) ** 1 + 1))
    return ηAE

# ET1区域热力学吸入
def cal_th_in_ET1(D, V, Fn, SF_t):
    ηTH = 0.5 * (1 - np.exp(-18 * (D * (V * Fn * SF_t) ** -0.25) ** 0.5))
    return ηTH

# ET2区域空气动力学吸入
def cal_ae_in_ET2(d, V, Fn, SF_t):
    ηAE = 1 - 1 / (5.50E-05 * (d ** 2 * V * Fn * SF_t ** 3) ** 1.17 + 1)
    return ηAE

# ET2区域热力学吸入
def cal_th_in_ET2(D, V, Fn, SF_t):
    ηTH = 1 - np.exp(-15.1 * (D * (V * Fn * SF_t) ** -0.25) ** 0.538)
    return ηTH

def cal_thermo_correction_factor(d):
    ψTH = 1 + 100 * np.exp(-1 * (np.log10(100 + 10 / d ** 0.9)) ** 2)
    return ψTH

def cal_φB(VD_ET, Vt):
    φB = 1 - VD_ET / Vt
    return φB

def cal_TB_timeconst(VD_BB, Vt, FRC, V, φB):
    tB = VD_BB * (1 + 0.5 * Vt / FRC) / (V * φB)
    return tB

# BB区域空气动力学吸入
def cal_ae_in_BB(d, V, Fn, SF_t):
    ηAE = 1 - np.exp(-4.08E-06 * (d ** 2 * V * Fn * SF_t ** 2.3) ** 1.152)
    return ηAE

# BB区域热力学吸入
def cal_th_in_BB(SF_t, ψth, D, tB):
    ηTH = 1 - np.exp(-22.02 * SF_t ** 1.24 * ψth * (D * tB) ** 0.6391)
    return ηTH

def cal_φb(VD_ET, VD_BB, Vt):
    φb = 1 - (VD_ET + VD_BB) / Vt
    return φb

def cal_bb_timeconst(VD_bb, Vt, FRC, V, φb):
    tb = VD_bb * (1 + 0.5 * Vt / FRC) / (V * φb)
    return tb

# bb区域空气动力学吸入
def cal_ae_in_bb(tb, d):
    ηAE = 1 - np.exp(-0.1147 * ((0.056 + tb ** 1.5) * d ** tb ** -0.29) ** 1.173)
    return ηAE

# bb区域热力学吸入
def cal_th_in_bb(SF_b, D, tb):
    ηTH = 1 - np.exp((-1) * (-76.8 + 167 * SF_b ** 0.65) * (D * tb) ** 0.5676)
    return ηTH

def cal_φAI(VD_ET, VD_BB, VD_bb, Vt):
    φAI = 1 - (VD_ET + VD_BB + VD_bb) / Vt
    return φAI

def cal_AI_timeconst(Vt, VD_ET, VD_BB, VD_bb, FRC, V):
    tAI = (Vt - VD_ET - (VD_BB + VD_bb) * (1 + Vt / FRC)) / V
    return tAI

# AI区域空气动力学吸入
def cal_ae_in_AI(SF_a, d, tAI):
    ηAE = 1 - np.exp((-0.146 * SF_a ** 0.98) * (d ** 2 * tAI) ** 0.6495)
    return ηAE

# AI区域热力学吸入
def cal_th_in_AI(SF_a, D, tAI):
    ηTH = 1 - np.exp(-(170 + 103 * SF_a ** 2.13) * (D * tAI) ** 0.6101)
    return ηTH


# bb区域空气动力学呼出
def cal_ae_ex_bb(tb, d):
    ηAE = 1 - np.exp(-0.1147 * ((0.056 + tb ** 1.5) * d ** tb ** -0.29) ** 1.173)
    return ηAE

# bb区域热力学呼出
def cal_th_ex_bb(SF_b, D, tb):
    ηTH = 1 - np.exp((-1) * (-76.8 + 167 * SF_b ** 0.65) * (D * tb) ** 0.5676)
    return ηTH

# BB区域空气动力学呼出
def cal_ae_ex_BB(d, V, Fn, SF_t):
    ηAE = 1 - np.exp(-2.04E-06 * (d ** 2 * V * Fn * SF_t ** 2.3) ** 1.152)
    return ηAE

# BB区域热力学呼出
def cal_th_ex_BB(SF_t, ψth, D, tB):
    ηTH = 1 - np.exp(-22.02 * SF_t ** 1.24 * ψth * (D * tB) ** 0.6391)
    return ηTH

# ET2区域空气动力学呼出
def cal_ae_ex_ET2(d, V, Fn, SF_t):
    ηAE = 1 - 1 / (5.50E-05 * (d ** 2 * V * Fn * SF_t ** 3) ** 1.17 + 1)
    return ηAE

# ET2区域热力学呼出
def cal_th_ex_ET2(D, V, Fn, SF_t):
    ηTH = 1 - np.exp(-15.1 * (D * (V * Fn * SF_t) ** -0.25) ** 0.538)
    return ηTH

# ET1区域空气动力学呼出
def cal_ae_ex_ET1(d, V, Fn, SF_t):
    ηAE = 0.5 * (1 - 1 / (0.0003 * (d ** 2 * V * Fn * SF_t) ** 1 + 1))
    return ηAE

# ET1区域热力学呼出
def cal_th_ex_ET1(D, V, Fn, SF_t):
    ηTH = 0.5 * (1 - np.exp(-18 * (D * (V * Fn * SF_t) ** -0.25) ** 0.5))
    return ηTH

def precompute_aux(base: BaseParams, beh: BehaviorParams):
        φB = cal_φB(base.VD_ET, beh.Vt)
        tB = cal_TB_timeconst(base.VD_BB, beh.Vt, base.FRC, beh.V, φB)
        φb = cal_φb(base.VD_ET, base.VD_BB, beh.Vt)
        tb = cal_bb_timeconst(base.VD_bb, beh.Vt, base.FRC, beh.V, φb)
        φAI = cal_φAI(base.VD_ET, base.VD_BB, base.VD_bb, beh.Vt)
        tAI = cal_AI_timeconst(beh.Vt, base.VD_ET, base.VD_BB, base.VD_bb, base.FRC, beh.V)
        #thermo = cal_thermo_correction_factor(dth)  # 若需 ψTH(d) 就在区域里调用

        thermo = cal_thermo_correction_factor
        
        return dict(
        φB=np.asarray(φB),
        φb=np.asarray(φb),
        φAI=np.asarray(φAI),
        tB=np.asarray(tB),
        tb=np.asarray(tb),
        tAI=np.asarray(tAI),
        thermo=thermo,           # ← 不要 np.array(...)
    )

def kernels_one_state(
        dae: np.ndarray, dth: np.ndarray,
        base: BaseParams, beh: BehaviorParams, *,
        nose_breath: bool = True, U_user: float = 1.0, ua: dict| None = None
) -> Dict[str, Dict[str, np.ndarray]]:
    import numpy as np
    ua = ua or {}
    # 基本向量与尺寸
    dae  = np.asarray(dae,  float).reshape(-1)   # (N_dae,)
    dth  = np.asarray(dth,  float).reshape(-1)   # (N_dth,)
    N_dae, N_dth = len(dae), len(dth)

    # 便捷扩维：AE(dae)->(N_dae,1), TH(dth)->(1,N_dth)
    AE = lambda v: np.asarray(v, float).reshape(-1)[:, None]
    TH = lambda v: np.asarray(v, float).reshape(-1)[None, :]

    # 吸入量，需能与 2D 核广播，统一成 (N_dae,1)
    inhal = AE(cal_in(dae, U_user))             # (N_dae,1)
    # 选择口/鼻呼吸
    Fn = beh.Fn_normal if nose_breath else beh.Fn_mouth
    if "Fn" in ua:
        Fn = ua["Fn"]
        
    aux = precompute_aux(base, beh)

    # 以 dth 为自变量的量，先算出 D_th、phi_th，然后 TH() 扩维
    D_th   = TH(CD(Cc(dth), dth))               # (1,N_dth)
    phi_th = TH(aux["thermo"](dth))             # (1,N_dth)

    # ================= 吸气：区域效率（拆 AE / TH） =================

    # ET1
    ae_in_ET1 = AE(cal_ae_in_ET1(dae, beh.V, Fn, base.SF_t))      # (N_dae,1)
    ae_in_ET1 *= ua.get("Cae_ET1", 1.0)

    th_in_ET1 = TH(cal_th_in_ET1(D_th.squeeze(), beh.V, Fn, base.SF_t))      # (1,N_dth) 传原 dth，或若函数需 D_th 就传 D_th.squeeze()
    th_in_ET1 *= ua.get("Cth_ET1", 1.0)
    # 若 cal_th_in_ET1 需要 D_th，请改为：TH(cal_th_in_ET1(D_th.squeeze(), beh.V, Fn, base.SF_t))
    eta_in_ET1    = np.sqrt(ae_in_ET1**2 + th_in_ET1**2)          # (N_dae,N_dth)
    dep_in_ET1    = eta_in_ET1 * inhal
    dep_in_ET1_AE = ae_in_ET1 * inhal
    dep_in_ET1_TH = th_in_ET1 * inhal

    # ET2
    ae_in_ET2 = AE(cal_ae_in_ET2(dae, beh.V, Fn, base.SF_t))
    ae_in_ET2 *= ua.get("Cae_ET2", 1.0)

    th_in_ET2 = TH(cal_th_in_ET2(D_th.squeeze(), beh.V, Fn, base.SF_t))
    th_in_ET2 *= ua.get("Cth_ET2", 1.0)
    # 若函数需 D_th，同上改为 D_th.squeeze()
    eta_in_ET2    = np.sqrt(ae_in_ET2**2 + th_in_ET2**2)
    up_ET2        = inhal * (1 - eta_in_ET1)                      # (N_dae,N_dth)
    dep_in_ET2    = eta_in_ET2 * up_ET2
    dep_in_ET2_AE = ae_in_ET2 * up_ET2
    dep_in_ET2_TH = th_in_ET2 * up_ET2

    # BB
    ae_in_BB = AE(cal_ae_in_BB(dae, beh.V, Fn, base.SF_t))
    ae_in_BB *= ua.get("Cae_BB", 1.0)

    th_in_BB = TH(cal_th_in_BB(base.SF_t, phi_th.squeeze(), D_th.squeeze(), aux["tB"]))  # 若 cal_th_in_BB 直接收 dth/phi_th
    th_in_BB *= ua.get("Cth_BB", 1.0)
    # 如果 cal_th_in_BB 需要 D_th（而非 dth），换成 D_th.squeeze()
    eta_in_BB    = np.sqrt(ae_in_BB**2 + th_in_BB**2)
    path_BB_in   = up_ET2 * (1 - eta_in_ET2) * aux["φB"]
    dep_in_BB    = eta_in_BB * path_BB_in
    dep_in_BB_AE = ae_in_BB * path_BB_in
    dep_in_BB_TH = th_in_BB * path_BB_in

    # bb
    ae_in_bb = AE(cal_ae_in_bb(aux["tb"], dae))
    ae_in_bb *= ua.get("Cae_bb", 1.0)

    th_in_bb = TH(cal_th_in_bb(base.SF_b, D_th.squeeze(), aux["tb"]))        # 或传 D_th.squeeze()，看你的函数签名
    th_in_bb *= ua.get("Cth_bb", 1.0)

    eta_in_bb    = np.sqrt(ae_in_bb**2 + th_in_bb**2)
    path_bb_in   = up_ET2 * (1 - eta_in_ET2) * (1 - eta_in_BB) * aux["φb"]
    dep_in_bb    = eta_in_bb * path_bb_in
    dep_in_bb_AE = ae_in_bb * path_bb_in
    dep_in_bb_TH = th_in_bb * path_bb_in

    # AI
    ae_in_AI = AE(cal_ae_in_AI(base.SF_a, dae, aux["tAI"]))
    ae_in_AI *= ua.get("Cae_AI", 1.0)

    th_in_AI = TH(cal_th_in_AI(base.SF_a, D_th.squeeze(), aux["tAI"]))       # 或传 D_th.squeeze()
    th_in_AI *= ua.get("Cth_AI", 1.0)
    
    eta_in_AI    = np.sqrt(ae_in_AI**2 + th_in_AI**2)
    path_AI_in   = up_ET2 * (1 - eta_in_ET2) * (1 - eta_in_BB) * (1 - eta_in_bb) * aux["φAI"]
    dep_in_AI    = eta_in_AI * path_AI_in
    dep_in_AI_AE = ae_in_AI * path_AI_in
    dep_in_AI_TH = th_in_AI * path_AI_in

    # 吸气合计
    dep_in_sum = dep_in_ET1 + dep_in_ET2 + dep_in_BB + dep_in_bb + dep_in_AI  # (N_dae,N_dth)

    # ================= 呼气（同样保持二维形状） =================
    common_ex = inhal * (1 - eta_in_ET1) * (1 - eta_in_ET2) * (1 - eta_in_BB) * (1 - eta_in_bb) * (1 - eta_in_AI)

    # bb（呼）
    ae_ex_bb = ae_in_bb; th_ex_bb = th_in_bb; eta_ex_bb = eta_in_bb
    dep_ex_bb    = common_ex * eta_ex_bb * aux["φb"]
    dep_ex_bb_AE = common_ex * ae_ex_bb * aux["φb"]
    dep_ex_bb_TH = common_ex * th_ex_bb * aux["φb"]

    # BB（呼）
    ae_ex_BB = AE(cal_ae_ex_BB(dae, beh.V, Fn, base.SF_t))
    th_ex_BB = TH(cal_th_ex_BB(base.SF_t, phi_th.squeeze(), D_th.squeeze(), aux["tB"]))  # 或 D_th.squeeze()
    eta_ex_BB  = np.sqrt(ae_ex_BB**2 + th_ex_BB**2)
    path_BB_ex = common_ex * (1 - eta_ex_bb)
    dep_ex_BB    = path_BB_ex * eta_ex_BB * aux["φB"]
    dep_ex_BB_AE = path_BB_ex * ae_ex_BB * aux["φB"]
    dep_ex_BB_TH = path_BB_ex * th_ex_BB * aux["φB"]

    # ET2（呼）
    ae_ex_ET2 = ae_in_ET2; th_ex_ET2 = th_in_ET2; eta_ex_ET2 = eta_in_ET2
    path_ET2_ex = path_BB_ex * (1 - eta_ex_BB)
    dep_ex_ET2    = path_ET2_ex * eta_ex_ET2
    dep_ex_ET2_AE = path_ET2_ex * ae_ex_ET2
    dep_ex_ET2_TH = path_ET2_ex * th_ex_ET2

    # ET1（呼）
    ae_ex_ET1 = ae_in_ET1; th_ex_ET1 = th_in_ET1; eta_ex_ET1 = eta_in_ET1
    path_ET1_ex = path_ET2_ex * (1 - eta_ex_ET2)
    dep_ex_ET1    = path_ET1_ex * eta_ex_ET1
    dep_ex_ET1_AE = path_ET1_ex * ae_ex_ET1
    dep_ex_ET1_TH = path_ET1_ex * th_ex_ET1

    # 区域合计（吸+呼）
    dep_sum_ET1 = dep_in_ET1 + dep_ex_ET1
    dep_sum_ET2 = dep_in_ET2 + dep_ex_ET2
    dep_sum_BB  = dep_in_BB  + dep_ex_BB
    dep_sum_bb  = dep_in_bb  + dep_ex_bb
    dep_sum_AI  = dep_in_AI  # AI 无呼气项

    # 返回
    return {
        "in":  {"ET1": dep_in_ET1, "ET2": dep_in_ET2, "BB": dep_in_BB, "bb": dep_in_bb, "AI": dep_in_AI},
        "ex":  {"ET1": dep_ex_ET1, "ET2": dep_ex_ET2, "BB": dep_ex_BB, "bb": dep_ex_bb},
        "sum": {"ET1": dep_sum_ET1, "ET2": dep_sum_ET2, "BB": dep_sum_BB, "bb": dep_sum_bb, "AI": dep_sum_AI},
        "components": {
            "ET1": {
                "AE_in": dep_in_ET1_AE, "TH_in": dep_in_ET1_TH,
                "AE_ex": dep_ex_ET1_AE, "TH_ex": dep_ex_ET1_TH,
                "AE": dep_in_ET1_AE + dep_ex_ET1_AE,
                "TH": dep_in_ET1_TH + dep_ex_ET1_TH,
                "TOTAL": dep_sum_ET1
            },
            "ET2": {
                "AE_in": dep_in_ET2_AE, "TH_in": dep_in_ET2_TH,
                "AE_ex": dep_ex_ET2_AE, "TH_ex": dep_ex_ET2_TH,
                "AE": dep_in_ET2_AE + dep_ex_ET2_AE,
                "TH": dep_in_ET2_TH + dep_ex_ET2_TH,
                "TOTAL": dep_sum_ET2
            },
            "BB": {
                "AE_in": dep_in_BB_AE, "TH_in": dep_in_BB_TH,
                "AE_ex": dep_ex_BB_AE, "TH_ex": dep_ex_BB_TH,
                "AE": dep_in_BB_AE + dep_ex_BB_AE,
                "TH": dep_in_BB_TH + dep_ex_BB_TH,
                "TOTAL": dep_sum_BB
            },
            "bb": {
                "AE_in": dep_in_bb_AE, "TH_in": dep_in_bb_TH,
                "AE_ex": dep_ex_bb_AE, "TH_ex": dep_ex_bb_TH,
                "AE": dep_in_bb_AE + dep_ex_bb_AE,
                "TH": dep_in_bb_TH + dep_ex_bb_TH,
                "TOTAL": dep_sum_bb
            },
            "AI": {
                "AE_in": dep_in_AI_AE, "TH_in": dep_in_AI_TH,
                "AE_ex": np.zeros_like(dep_in_AI_AE), "TH_ex": np.zeros_like(dep_in_AI_TH),
                "AE": dep_in_AI_AE, "TH": dep_in_AI_TH,
                "TOTAL": dep_sum_AI
            }
        }
    }