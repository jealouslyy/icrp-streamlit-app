from __future__ import annotations
from typing import Dict, Mapping, Optional, List, Union
import numpy as np
from scipy.stats import lognorm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import BaseParams, BehaviorParams

try:
    from .params import BaseParams, BehaviorParams
except Exception:
    BaseParams = object
    BehaviorParams = object

from .model import kernels_one_state

REGIONS = ("ET1","ET2","BB","bb","AI")

# 放在 core/dose.py 内，_get_dep_curves 定义的附近即可
def _align_dep_curves_to_dae(dep_dict, dae, dth, *, src_name: str):
    """
    把 _get_dep_curves 返回的各区数组，统一成与 dae 等长的一维曲线。
    支持输入为 2D kernel:(len(dae), len(dth)) 或 (len(dth), len(dae))，会沿 dth 轴取均值；
    输入为 1D 时必须长度等于 len(dae)；否则明确报错。
    """
    import numpy as np
    dae = np.asarray(dae, float).reshape(-1)
    len_dae = len(dae)
    len_dth = len(dth) if dth is not None else None

    out = {}
    for r, arr in dep_dict.items():
        A = np.asarray(arr, float)
        if A.ndim == 2:
            if len_dth is None:
                raise ValueError(f"[{src_name}] {r} 给了 2D 核但 dth=None，无法判断聚合方向")
            if A.shape == (len_dae, len_dth):
                frac = A.mean(axis=1)             # -> (len_dae,)
            elif A.shape == (len_dth, len_dae):
                frac = A.mean(axis=0)             # 轴颠倒，纠正
            else:
                raise ValueError(
                    f"[{src_name}] {r} 2D kernel 形状 {A.shape} 与 "
                    f"(len(dae),len(dth))=({len_dae},{len_dth}) 不匹配"
                )
        elif A.ndim == 1:
            if A.shape[0] == len_dae:
                frac = A
            elif (len_dth is not None) and (A.shape[0] == len_dth):
                # 常见误用：把 dth 方向当成了 dae 方向
                raise ValueError(
                    f"[{src_name}] {r} 得到长度 {A.shape[0]} 的 1D 数组（像是 dth 轴），"
                    f"应先在 dth 上聚合后得到与 dae 等长的曲线（{len_dae}）"
                )
            else:
                raise ValueError(
                    f"[{src_name}] {r} 1D 曲线长度 {A.shape[0]} 不等于 len(dae)={len_dae}"
                )
        else:
            raise ValueError(f"[{src_name}] {r} kernel 维度 {A.ndim} 非法")
        out[r] = np.clip(frac, 0.0, 1.0)
    return out


def _get_dep_curves(
    dae: np.ndarray,
    dth: Optional[np.ndarray],
    base: "BaseParams",
    beh: "BehaviorParams", *,
    nose_breath: bool,
    wind_speed: float
) -> Dict[str, np.ndarray]:
    """
    调用内核，拿“实际沉积分布曲线”（吸+呼合计），随 dae 变化。
    返回形如 {'ET1':arr, 'ET2':arr, 'BB':arr, 'bb':arr, 'AI':arr}
    """
    K = kernels_one_state(
        dae=dae, dth=dth, base=base, beh=beh,
        nose_breath=nose_breath, U_user=wind_speed
    )
    return K["sum"]

# 1) 离散：直径 + 浓度（Eq.2，直接用 B）
# ------------------------------------------------------------
def dose_discrete_one_behavior(
    dae: np.ndarray,
    conc_ug_m3: np.ndarray,
    dth: Optional[np.ndarray],
    base: BaseParams,
    beh: BehaviorParams, *,
    duration_h: float = 1.0,
    nose_breath: bool,
    wind_speed: float,
) -> Dict[str, float]:
    import numpy as np

    dep_raw = _get_dep_curves(dae, dth, base, beh, nose_breath=nose_breath, wind_speed=wind_speed)
    dep = _align_dep_curves_to_dae(dep_raw, dae, dth, src_name="dose_discrete_one_behavior")

    dae = np.asarray(dae, float).reshape(-1)
    C   = np.asarray(conc_ug_m3, float).reshape(-1)
    if C.shape != dae.shape:
        raise ValueError(f"[dose_discrete_one_behavior] conc 形状 {C.shape} 与 dae 形状 {dae.shape} 不一致")

    B = float(getattr(beh, "B", 0.0))
    t = float(duration_h)

    out: Dict[str, float] = {}
    total = 0.0

    # DEBUG：命令行可见
    print(f"[DEBUG one_behavior] dae={dae.shape}, conc={C.shape}, B={B}, t={t}")

    for r in REGIONS:
        n_curve = np.asarray(dep[r], float).reshape(-1)

        # 关键：在真正相乘前给出可读错误
        try:
            _ = C * n_curve
        except Exception as e:
            import numpy as np
            raise ValueError(
                f"[dose_discrete_one_behavior] 区域 {r} 广播失败："
                f"C.shape={np.shape(C)}, n_curve.shape={np.shape(n_curve)}；"
                f"len(dae)={len(dae)}, len(dth)={len(dth) if dth is not None else 'None'}。\n"
                f"很可能 n_curve 是按 dth(=300) 而不是 dae(=len(conc)) 计算的。"
            ) from e

        H = float(np.sum(B * C * n_curve)) * t

        # 如需按粒径积分，可以用：H = float(np.trapz(B * C * n_curve, x=dae)) * 1.0
        out[r] = H
        total += H

    out["TOTAL"] = total
    return out

def dose_discrete_multi_behaviors(
    dae: np.ndarray,
    conc_ug_m3: np.ndarray,
    dth: Optional[np.ndarray],
    base: BaseParams,
    behaviors: Mapping[str, BehaviorParams],
    durations_h: Mapping[str, float], *,
    nose_breath: bool,
    wind_speed: float,
) -> Dict[str, Dict[str, float]]:
    """
    多行为 + 各自时长：逐行为按 Eq.(2) 计算再相加
      H_j(total) = Σ_b  Σ_i [ B_b * C_i * n_{i,j}(b) ] * t_b
    返回：
      {
        'by_behavior': {行为: {区:µg,...,'TOTAL':µg}},
        'sum_by_region': {区:µg,...,'TOTAL':µg},
        'sum_by_behavior': {行为:µg,...,'TOTAL':µg}
      }
    """
    by_behavior: Dict[str, Dict[str, float]] = {}
    sum_by_region: Dict[str, float] = {r: 0.0 for r in REGIONS}
    sum_by_region["TOTAL"] = 0.0
    sum_by_behavior: Dict[str, float] = {}

    for name, beh in behaviors.items():
        t = float(durations_h.get(name, 0.0))
        if t <= 0:
            continue
        Hj = dose_discrete_one_behavior(
            dae=dae, conc_ug_m3=conc_ug_m3, dth=dth,
            base=base, beh=beh, duration_h=t,
            nose_breath=nose_breath, wind_speed=wind_speed,
        )

        # ---- 调试信息 & 形状检查 ----
        import numpy as np
        print(f"[DEBUG] 行为={name}, dae形状={np.shape(dae)}, conc形状={np.shape(conc_ug_m3)}")
        for r in REGIONS:
            if r in Hj:
                arr = np.asarray(Hj[r])
                if arr.shape != dae.shape and arr.shape != ():  # () 是单值
                    raise ValueError(
                        f"[dose_discrete_multi_behaviors] 区域 {r} 的数组形状 {arr.shape} "
                        f"与 dae 形状 {dae.shape} 不一致"
                    )

        by_behavior[name] = Hj
        sum_by_behavior[name] = Hj["TOTAL"]
        for r in REGIONS:
            sum_by_region[r] += Hj[r]
        sum_by_region["TOTAL"] += Hj["TOTAL"]

    sum_by_behavior["TOTAL"] = sum_by_region["TOTAL"]
    return {
        "by_behavior": by_behavior,
        "sum_by_region": sum_by_region,
        "sum_by_behavior": sum_by_behavior,
    }

# ------------------------------------------------------------
# 2) 只有直径（无浓度）：先得到“分数沉积”，再用平均浓度与吸入体积换算
#    这里仍沿用 B，但以 Σ(B_b * t_b) 表示总吸入体积 (m³)
# ------------------------------------------------------------
def dose_discrete_one_behavior(
    dae: np.ndarray,
    conc_ug_m3: np.ndarray,
    dth: Optional[np.ndarray],
    base: BaseParams,
    beh: BehaviorParams, *,
    duration_h: float = 1.0,
    nose_breath: bool,
    wind_speed: float,
) -> Dict[str, float]:
    import numpy as np

    dep_raw = _get_dep_curves(dae, dth, base, beh, nose_breath=nose_breath, wind_speed=wind_speed)
    dep = _align_dep_curves_to_dae(dep_raw, dae, dth, src_name="dose_discrete_one_behavior")

    dae = np.asarray(dae, float).reshape(-1)
    C   = np.asarray(conc_ug_m3, float).reshape(-1)
    if C.shape != dae.shape:
        raise ValueError(f"[dose_discrete_one_behavior] conc 形状 {C.shape} 与 dae 形状 {dae.shape} 不一致")

    B = float(getattr(beh, "B", 0.0))
    t = float(duration_h)

    out: Dict[str, float] = {}
    total = 0.0

    # DEBUG：命令行可见
    print(f"[DEBUG one_behavior] dae={dae.shape}, conc={C.shape}, B={B}, t={t}")

    for r in REGIONS:
        n_curve = np.asarray(dep[r], float).reshape(-1)

        try:
            _ = C * n_curve
        except Exception as e:
            raise ValueError(
                f"[dose_discrete_one_behavior] 区域 {r} 广播失败："
                f"C.shape={C.shape}, n_curve.shape={n_curve.shape}, "
                f"len(dae)={len(dae)}, len(dth)={len(dth) if dth is not None else 'None'}"
            ) from e

        H = float(np.sum(B * C * n_curve)) * t
        # 如需按粒径积分，可以用：H = float(np.trapz(B * C * n_curve, x=dae)) * 1.0
        out[r] = H
        total += H

    out["TOTAL"] = total
    return out


def dose_from_fraction_with_Cmean_and_B(
    DE_frac: Mapping[str, float],
    behaviors: Mapping[str, BehaviorParams],
    durations_h: Mapping[str, float],
    C_mean_ug_m3: float
) -> Dict[str, float]:
    """
    剂量 = 平均浓度 * 总吸入体积 * 分数
         = C_mean * [ Σ_b (B_b * t_b) ] * DE_j
    返回 {区域: µg, 'TOTAL': µg}
    """
    V_inh = 0.0  # m³
    for name, beh in behaviors.items():
        t = float(durations_h.get(name, 0.0))
        if t > 0:
            V_inh += float(getattr(beh, "B", 0.0)) * t  # B(m³/h)*小时 = m³

    out: Dict[str, float] = {}
    total = 0.0
    for r in REGIONS:
        H = float(C_mean_ug_m3) * float(V_inh) * float(DE_frac[r])
        out[r] = H
        total += H
    out["TOTAL"] = total
    return out

# ------------------------------------------------------------
# 3) 多分散：对数正态 PDF 加权（分数→剂量）
# ------------------------------------------------------------
def fractional_dep_from_lognormal(
    dae: np.ndarray,
    dth: Optional[np.ndarray],
    base: BaseParams,
    beh: BehaviorParams, *,
    nose_breath: bool,
    wind_speed: float,
    AMAD: float,
    sigma_AD: float,
    AMTD: float,
    sigma_TD: float
) -> Dict[str, np.ndarray]:
    """
    用 p_ae(dae) 对“实际沉积曲线”做式(18)积分，得到各区分数。
    返回：每个区域 -> ndarray(0D 或 1D),'TOTAL' 为逐元素相加。
    """
    # 1) 取各区“实际沉积分数曲线”（随 dae）
    dep = _get_dep_curves(dae, dth, base, beh,
                          nose_breath=nose_breath, wind_speed=wind_speed)

    # 2) p_ae(dae)：对数正态，沿 dae 归一
    dae = np.asarray(dae, float).reshape(-1)
    p_ae = lognorm.pdf(dae, s=np.log(sigma_AD), scale=AMAD)
    Z = np.trapz(p_ae, x=dae)
    if Z > 0:
        p_ae = p_ae / Z

    # 3) 沿直径轴积分（支持 1D 或 2D 曲线；结果是 0D 或 1D）
    DE: Dict[str, np.ndarray] = {}
    for r in REGIONS:
        curve = np.asarray(dep[r], float)              # (n_d,) 或 (n_k, n_d)
        # --- 关键修补：若曲线是百分比(0-100)，转成分数(0-1) ---
        if np.nanmax(curve) > 1.05:                    # 留一点浮动余量
            curve = curve / 100.0
        # ------------------------------------------------------
        val = np.trapz(curve * p_ae, x=dae, axis=-1)   # -> (n_k,) 或 标量
        DE[r] = np.asarray(val, float)

    # 4) TOTAL 逐元素求和（向量安全）
    DE["TOTAL"] = np.sum([DE[r] for r in REGIONS], axis=0)
    return DE

def fractional_dep_from_discrete_bins(
    dae: np.ndarray,
    dth: Optional[np.ndarray],
    base: BaseParams,
    beh: BehaviorParams, *,
    nose_breath: bool,
    wind_speed: float,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    在离散 dae 网格上对 dep(dae) 做加权积分，得到各区分数 DE_j。
    """
    import numpy as np

    dep_raw = _get_dep_curves(dae, dth, base, beh,
                              nose_breath=nose_breath,
                              wind_speed=wind_speed)
    dep = _align_dep_curves_to_dae(dep_raw, dae, dth,
                                   src_name="fractional_dep_from_discrete_bins")

    dae = np.asarray(dae, float).reshape(-1)
    w = np.ones_like(dae, float) if weights is None else np.asarray(weights, float).reshape(-1)
    if w.shape != dae.shape:
        raise ValueError("weights 长度必须与 dae 相同")

    Z = np.trapz(w, dae)
    if Z > 0:
        w = w / Z

    DE: Dict[str, float] = {}
    for r in REGIONS:
        curve = dep[r]   # 已与 dae 对齐
        DE[r] = float(np.trapz(curve * w, dae))
    DE["TOTAL"] = float(sum(DE[r] for r in REGIONS))
    return DE

def dose_from_lognormal_with_B(
    DE_frac: Mapping[str, Union[float, List[float], np.ndarray]],
    behaviors: Mapping[str, "BehaviorParams"],
    durations_h: Mapping[str, float],
    C_mean_ug_m3: float
) -> Dict[str, Union[float, List[float]]]:
    """
    多分散: Dose_j = C_mean * [ Σ_b (B_b * t_b) ] * DE_j
    - 单分散:DE_j 为 float -> 返回 float
    - 多分散:DE_j 为 list/array -> 返回 list(逐样本结果)
    """
    # 吸入体积系数 Σ(B_b * t_b)
    V_inh = 0.0
    for name, beh in behaviors.items():
        t = float(durations_h.get(name, 0.0))
        if t > 0:
            V_inh += float(getattr(beh, "B", 0.0)) * t
    
    print(f"[DEBUG] Σ(B*t)=V_inh={V_inh:.3f} m³ (常见量级:0.2~3 m³;明显大于此请检查单位)")
    coeff = float(C_mean_ug_m3) * float(V_inh)

    out: Dict[str, Union[float, List[float]]] = {}
    total_acc: Union[float, np.ndarray, None] = None

    for r in REGIONS:
        val = np.asarray(DE_frac[r], dtype=float)    # 标量或向量都兼容
        H = coeff * val                              # 按元素计算剂量

        if H.ndim == 0:                              # 标量路径
            out[r] = H.item()
            total_acc = (0.0 if total_acc is None else total_acc) + out[r]
        else:                                        # 向量路径
            out[r] = H.tolist()
            total_acc = (np.zeros_like(H) if total_acc is None else total_acc) + H

    out["TOTAL"] = total_acc if isinstance(total_acc, float) else np.asarray(total_acc).tolist()
    return out
