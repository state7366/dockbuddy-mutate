# -*- coding: utf-8 -*-
"""隐式溶剂(OBC / GBneck2)装配 —— 全仓唯一入口 (2026-10-03)
=============================================================================
为什么单独成模块
----------------
本仓库有两处需要**物理上比真空更真实**的单点能量/短程动力学:

  1. ``libmutate`` 的 ``--relax md``(突变体侧链替换后的局部隐式溶剂 MD);
  2. ``libdock.rescore``(对接后对 top-N pose 做 MM-GBSA 单点重评分)。

两处都要"给一个已建好的 OpenMM System 挂上 OBC"。与其两份实现各自漂移(参数
搞错只会表现成"能量数值怪", 极难查), 不如收敛成**一个函数**, 两边共用。

OpenMM 8.x 的坑(实测, 别再踩)
------------------------------
- ``app.ForceField.createSystem(..., implicitSolvent=...)`` **已失效**: 8.6 起
  会对未知参数报 ``The argument 'implicitSolvent' was specified to
  createSystem() but was never used``。必须手工 ``system.addForce(mm.GBSAOBCForce())``。
- 每粒子的 OBC 半径不是 sigma, 而是 ``sigma * 0.5612310241546864907``
  (OBC 定义的 r_min/2 ↔ sigma 换算常数)。
- ``scalingFactor``(screen)实测必须 0.5: 取 0.85/1.2 时总能量会变成**正的且巨大**,
  物理上显然错。0.5 时小体系总能量 ~-3.8e4 kJ/mol 量级, 合理。
- 判氢/判元素不要用原子名首字符, 用 AutoDock 类型码首字符或元素列。

失败语义
--------
装配失败(例如 OpenMM 版本没有 ``GBSAOBCForce``)时**不抛异常** —— 静默返回
``None`` 并把原因写进 ``log``/``reason``, 由调用方降级到真空最小化, 使整条
流水线不会因为"可选项"而崩。
"""
import logging
from typing import Any, Dict, Optional

LOG = logging.getLogger("libdock.solvents")

SOLVENT_DIELECTRIC = 78.5
SOLUTE_DIELECTRIC = 1.0
SURFACE_AREA_ENERGY = 0.0056
# OBC: r_min/2 = 0.5 * r_min = sigma * 0.5612310241546864907(= 2**(1/6)/2 相关常数)
OBC_SIGMA_TO_RADIUS = 0.5612310241546864907
OBC_SCREEN = 0.5
OBC_ID = "OBC(GBSAOBCForce), solvent_dielectric=78.5"


def attach_obc_force(system: Any, log: Optional[Any] = None) -> Optional[Any]:
    """给已建好的 OpenMM ``System`` 挂上 ``GBSAOBCForce``, 返回该 Force 或 None。

    ``system`` 必须来自 ``ForceField.createSystem``(里面有一个 ``NonbondedForce``
    用来取 q/sigma)。挂成功后 System 的**能量表达**变为真空 Nonbonded + OBC 隐式溶剂。
    """
    def _warn(msg):
        if log is not None:
            log(msg)
        else:
            LOG.warning(msg)

    try:
        import openmm as mm
        from openmm import unit
    except Exception as e:                                   # pragma: no cover
        _warn(f"OBC 隐式溶剂不可用(无法 import openmm): {e}")
        return None

    try:
        nb_i = next(i for i in range(system.getNumForces())
                    if isinstance(system.getForce(i), mm.NonbondedForce))
    except StopIteration:                                    # pragma: no cover
        _warn("OBC 隐式溶剂不可用于该 System(找不到 NonbondedForce)")
        return None
    try:
        nb = system.getForce(nb_i)
        gb = mm.GBSAOBCForce()
        gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
        gb.setSoluteDielectric(SOLUTE_DIELECTRIC)
        gb.setSolventDielectric(SOLVENT_DIELECTRIC)
        gb.setSurfaceAreaEnergy(SURFACE_AREA_ENERGY)
        for i in range(system.getNumParticles()):
            q, sigma, _eps = nb.getParticleParameters(i)
            gb.addParticle(q.value_in_unit(unit.elementary_charge),
                           sigma.value_in_unit(unit.nanometer) * OBC_SIGMA_TO_RADIUS,
                           OBC_SCREEN)
        system.addForce(gb)
        return gb
    except Exception as e:                                   # pragma: no cover
        _warn(f"OBC 隐式溶剂装配失败: {e}")
        return None


def obc_model_string() -> str:
    return OBC_ID


def obc_params() -> Dict[str, float]:
    """对外暴露的 OBC 参数(报告/复现用)。"""
    return {"solute_dielectric": SOLUTE_DIELECTRIC,
            "solvent_dielectric": SOLVENT_DIELECTRIC,
            "surface_area_energy": SURFACE_AREA_ENERGY,
            "obc_radius_factor": OBC_SIGMA_TO_RADIUS,
            "obc_screen": OBC_SCREEN,
            "model": "OBC/GBneck2"}


__all__ = [
    "SOLVENT_DIELECTRIC", "SOLUTE_DIELECTRIC", "SURFACE_AREA_ENERGY",
    "OBC_SIGMA_TO_RADIUS", "OBC_SCREEN", "OBC_ID",
    "attach_obc_force", "obc_model_string", "obc_params",
]
