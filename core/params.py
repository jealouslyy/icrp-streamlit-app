from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Literal
import json

Material = Literal["F", "M", "S"]

# ---------- 1) 数据结构 ----------
@dataclass
class BaseParams:
    FRC: float; VD_ET: float; VD_BB: float; VD_bb: float
    Height: float; d0: float; d9: float; d16: float
    SF_t: float; SF_b: float; SF_a: float

@dataclass
class BehaviorParams:
    Fn_normal: float          # 鼻呼吸比例或“正常呼吸系数”
    Fn_mouth: float           # 口呼比例
    B: float                  # 通气量倍率（或你用的行为系数）
    Fr: float                 # 次/分钟
    Vt: float                 # mL/次
    V: float                  # mL/分钟（）

#Population = Dict[str, object]  # {"base": BaseParams, "behaviors": Dict[str, BehaviorParams]}

# ---------- 2)  ----------
POP: Dict[str, Dict[str, object]] = {
    # ======= 30岁成年男性 =======
    "male_30y": {
        "base": BaseParams(FRC=3301, VD_ET=50, VD_BB=49, VD_bb=47, Height=176,
                           d0=1.65, d9=0.165, d16=0.051, SF_t=1, SF_b=1, SF_a=1),
        "behaviors": {
            "sleep":   BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.45, Fr=12, Vt=625,  V=250),
            "rest":    BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.54, Fr=12, Vt=750,  V=300),
            "light":   BehaviorParams(Fn_normal=1,   Fn_mouth=0.4, B=1.50, Fr=20, Vt=1250, V=833),
            "heavy":   BehaviorParams(Fn_normal=0.5, Fn_mouth=0.3, B=3.00, Fr=26, Vt=1920, V=1670),
        }
    },
    # ======= 15岁男性 =======
    "male_15y": {
        "base": BaseParams(FRC=2677, VD_ET=45, VD_BB=44, VD_bb=41, Height=169,
                           d0=1.59, d9=0.161, d16=0.047, SF_t=1.04, SF_b=1.03, SF_a=1.07),
        "behaviors": {
            "sleep": BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.42, Fr=14, Vt=500,  V=233),
            "rest":  BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.48, Fr=15, Vt=533,  V=267),
            "light": BehaviorParams(Fn_normal=1,   Fn_mouth=0.4, B=1.38, Fr=23, Vt=1000, V=767),
            "heavy": BehaviorParams(Fn_normal=0.5, Fn_mouth=0.3, B=2.92, Fr=36, Vt=1352, V=1622),
        }
    },
    # ======= 30岁成年女性 =======
    "female_30y": {
        "base": BaseParams(FRC=2681, VD_ET=40, VD_BB=40, VD_bb=44, Height=163,
                           d0=1.53, d9=0.159, d16=0.048, SF_t=1.08, SF_b=1.04, SF_a=1.07),
        "behaviors": {
            "sleep": BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.32, Fr=12, Vt=444,  V=178),
            "rest":  BehaviorParams(Fn_normal=1,   Fn_mouth=0.4, B=0.39, Fr=14, Vt=464,  V=217),
            "light": BehaviorParams(Fn_normal=1,   Fn_mouth=0.4, B=1.25, Fr=21, Vt=992,  V=694),
            "heavy": BehaviorParams(Fn_normal=0.5, Fn_mouth=0.3, B=2.70, Fr=33, Vt=1364, V=1500),
        }
    },
    # ======= 15岁女性 =======
    "female_15y": {
        "base": BaseParams(FRC=2325, VD_ET=39, VD_BB=39, VD_bb=37, Height=161,
                           d0=1.52, d9=0.156, d16=0.045, SF_t=1.09, SF_b=1.06, SF_a=1.13),
        "behaviors": {
            "sleep": BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.35, Fr=14, Vt=417,  V=194),
            "rest":  BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.40, Fr=16, Vt=417,  V=222),
            "light": BehaviorParams(Fn_normal=1,   Fn_mouth=0.4, B=1.30, Fr=24, Vt=903,  V=722),
            "heavy": BehaviorParams(Fn_normal=0.5, Fn_mouth=0.3, B=2.57, Fr=38, Vt=1127, V=1428),
        }
    },
    # ======= 10岁儿童 =======
    "child_10y": {
        "base": BaseParams(FRC=1484, VD_ET=25, VD_BB=26, VD_bb=26, Height=138,
                           d0=1.31, d9=0.143, d16=0.039, SF_t=1.26, SF_b=1.16, SF_a=1.31),
        "behaviors": {
            "sleep": BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.31, Fr=17, Vt=304, V=172),
            "rest":  BehaviorParams(Fn_normal=1,   Fn_mouth=0.7, B=0.38, Fr=19, Vt=333, V=211),
            "light": BehaviorParams(Fn_normal=1,   Fn_mouth=0.4, B=1.12, Fr=32, Vt=583, V=622),
            "heavy": BehaviorParams(Fn_normal=0.5, Fn_mouth=0.3, B=2.03, Fr=45, Vt=752, V=1128),
        }
    },
    # ======= 5岁儿童（重度运动缺省）=======
    "child_5y": {
        "base": BaseParams(FRC=767, VD_ET=13.3, VD_BB=15.5, VD_bb=16.7, Height=110,
                           d0=1.06, d9=0.127, d16=0.031, SF_t=1.55, SF_b=1.30, SF_a=1.63),
        "behaviors": {
            "sleep": BehaviorParams(Fn_normal=1, Fn_mouth=0.7, B=0.24, Fr=23, Vt=174, V=133),
            "rest":  BehaviorParams(Fn_normal=1, Fn_mouth=0.7, B=0.32, Fr=25, Vt=213, V=178),
            "light": BehaviorParams(Fn_normal=1, Fn_mouth=0.4, B=0.57, Fr=39, Vt=244, V=317),
            # "heavy": 缺省
        }
    },
}

# ---------------清除模型参数结构-----------------
@dataclass
class ClearanceTransferParams:
    """传输/清除速率(单位：天^-1)"""
    m1_4: float = 0.02
    m2_4: float = 0.001
    m3_4: float = 0.0001
    m3_10: float = 0.00002
    m4_7: float = 2.0
    m5_7: float = 0.03
    m6_10: float = 0.01
    m7_11: float = 10.0
    m8_11: float = 0.03
    m9_10: float = 0.01
    m11_15: float = 100.0
    m12_13: float = 0.001
    m14_16: float = 1.0

class ClearancePartitionParams:
    """分配比例（各室内子室权重 / 分流基数）"""
    P1: float = 0.3
    P2: float = 0.6
    P3: float = 0.1
    # bb/BB 主通道基数（最终会减去 fs）
    P4_base: float = 0.993
    P7_base: float = 0.993
    # 可溶分流基数（最终直接用 fs 覆盖）
    P5_base: float = 1.0
    P8_base: float = 1.0
    # 串联系统
    P6: float = 0.007
    P9: float = 0.007
    # ET2 主/次分流
    P11: float = 0.9995
    P12: float = 0.0005

@dataclass
class AbsorptionParams:
    """物质型态吸收参数"""
    fr: float   # toward blood fast fraction
    sr: float   # fast rate
    ss: float   # slow rate

# 物质型态 → 吸收参数 预设
CLEARANCE_ABSORB_PROFILES: Dict[Material, AbsorptionParams] = {
    "F": AbsorptionParams(fr=1.0,   sr=100.0, ss=0.0),
    "M": AbsorptionParams(fr=0.1,   sr=100.0, ss=0.005),
    "S": AbsorptionParams(fr=0.001, sr=100.0, ss=0.0001),
}

# 默认（全局）清除参数：如需按人群/场景覆盖，可在 POP 里挂载一个 "clearance" 字段
CLEARANCE_DEFAULTS = {
    "transfer":  ClearanceTransferParams(),
    "partition": ClearancePartitionParams(),
}
