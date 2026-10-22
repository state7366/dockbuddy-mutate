# -*- coding: utf-8 -*-
"""
libmutate.py —— 定点突变建模引擎抽象层 (2026-09-30 新增)
===================================================================================
动机: 单点突变不需要也不应重跑 AlphaFold(AF 对点突变不敏感, Buel & Walters 2022
NSMB: 突变体预测结构≈WT, backbone RMSD<1A)。正解 = 在现有 WT 结构上替换侧链
+ 局部松弛。骨架/坐标系不变 → 突变体可直接沿用 WT 的盒子与种子矩阵做对比对接。

架构: MutantEngine 抽象基类 + 注册表, 当前唯一实现为 PdbfixerOpenMMEngine;
FoldX/SCWRL4/MODELLER/PyMOL 为预留桩(is_available 检测环境, mutate 给出
明确的"接口预留"错误), 将来按同一接口补实现即可, 上层(mutate 阶段)零改动。

设计决策(记录备查):
  - 输出为**纯蛋白 PDB**(ATOM only): 共晶配体/水在突变体中没有确定的放置方式,
    保留反而误导; 配体依赖的盒子定位在突变体上应由 --inherit_box 继承 WT 盒子。
    被丢弃的 HETATM 记入报告 ligands_dropped 并打 WARNING。
  - **不加氢**: 与 DockBuddy 受体准备链路(openbabel 统一加氢/定质子化)保持单一
    职责, 避免两处加氢口径不一致。
  - **B 因子回填**: openmm PDBFile.writeFile 把 tempFactor 写成 0.00, 会丢失
    AF 模型的 pLDDT( DockBuddy 用 B 因子列做 pLDDT 口径)。mutate 后按
    (chain,resnum,atom) 把原 B 因子写回; 新生侧链原子取该残基原残基均值。
  - OpenMM 局部最小化: 真空力场 + 半径外原子谐振约束(默认 8A, k=1000
    kJ/mol/nm^2), 只放行突变位点邻域, 保证"局部松弛、全局不动"。
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("dockpipeline")

# 环境兜底: OpenMM 默认平台强制 CPU。
# 动机: 在 CUDA 环境混乱(多版本 toolkit、PATH 同时暴露 cudart64_12/13)的机器上,
# OpenMM 创建 Context 不显式指定平台时按速度自动探测(CUDA→OpenCL→CPU),
# 探测过程加载到不兼容插件会在原生层 fast-fail(Windows 0xC06D007F), Python 层无任何报错。
# 关键: 这不只影响我们自己的 _minimize_local(已显式 CPU),
# PDBFixer.addMissingAtoms() 内部也会创建【默认平台】的 Context, 无法从外部指定。
# OPENMM_DEFAULT_PLATFORM 在每次 Context 创建时被读取, 对所有 Context 生效;
# 用 setdefault —— 用户若显式设过该变量(如想用 GPU)则尊重其选择。
# 2026-10-08: 支持显式切 GPU —— 设 DOCK_OPENMM_PLATFORM=CUDA(或 OPENMM_DEFAULT_PLATFORM=CUDA)
# 即让本模块的显式 Context 也走 GPU(见 common.openmm_platform), 默认仍 CPU 保守。
os.environ.setdefault("OPENMM_DEFAULT_PLATFORM", "CPU")

# ----------------------------------------------------------------------------
# 氨基酸表
# ----------------------------------------------------------------------------
AA1_TO_3: Dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA3_TO_1: Dict[str, str] = {v: k for k, v in AA1_TO_3.items()}


# ----------------------------------------------------------------------------
# 突变规格
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MutSpec:
    """一个单点突变。chain 为空表示沿用调用处默认链。"""
    wt3: str
    resnum: int
    mut3: str
    chain: str = ""

    @property
    def label(self) -> str:
        return f"{AA3_TO_1.get(self.wt3, 'X')}{self.resnum}{AA3_TO_1.get(self.mut3, 'X')}"

    @property
    def pdbfixer(self) -> str:
        return f"{self.wt3}-{self.resnum}-{self.mut3}"


_MUT_1 = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])(?::([A-Za-z0-9]))?$")
_MUT_3 = re.compile(r"^([A-Za-z]{3})-?(\d+)-?([A-Za-z]{3})(?::([A-Za-z0-9]))?$")


def parse_mutation(text: str) -> MutSpec:
    """解析 'A123G' / 'A123G:B' / 'ALA-123-GLY' / 'ALA123GLY' 为 MutSpec。

    2026-10-03: 非标准氨基酸字母(X/B/J/O/U/Z 等)此前触发 AA1_TO_3 KeyError
    裸栈穿透到 cmd_mutate(其 except 只捕 ValueError/NotImplementedError),
    单例 `mutate --mut X56A` 直接崩 traceback —— 统一在源头转 ValueError。"""
    t = text.strip()
    m = _MUT_1.match(t)
    if m:
        wt1, num, mut1, ch = m.groups()
        for _c in (wt1, mut1):
            if _c.upper() not in AA1_TO_3:
                raise ValueError(f"非标准氨基酸字母 '{_c}'(突变 '{text}'): "
                                 f"仅支持 20 种标准氨基酸单字母")
        return MutSpec(AA1_TO_3[wt1.upper()], int(num), AA1_TO_3[mut1.upper()],
                       (ch or "").strip())
    m = _MUT_3.match(t)
    if m:
        wt3, num, mut3, ch = m.groups()
        for _r in (wt3, mut3):
            if _r.upper() not in AA3_TO_1:
                raise ValueError(f"非标准氨基酸 '{_r}'(突变 '{text}'): "
                                 f"仅支持 20 种标准氨基酸三字母")
        return MutSpec(wt3.upper(), int(num), mut3.upper(), (ch or "").strip())
    raise ValueError(f"无法解析突变 '{text}': 支持 A123G / A123G:B / ALA-123-GLY / ALA123GLY[:B]")


def parse_mutations(texts: Sequence[str], default_chain: str = "") -> List[MutSpec]:
    """解析突变列表(也接受逗号分隔)。空 chain 的一律补 default_chain。"""
    out: List[MutSpec] = []
    for t in texts:
        for piece in str(t).split(","):
            piece = piece.strip()
            if not piece:
                continue
            ms = parse_mutation(piece)
            if not ms.chain and default_chain:
                ms = MutSpec(ms.wt3, ms.resnum, ms.mut3, default_chain)
            out.append(ms)
    return out


# ----------------------------------------------------------------------------
# PDB 轻量解析 / Kabsch / clash(纯 Python, 供引擎 QC 与验证脚本共用)
# ----------------------------------------------------------------------------
@dataclass
class PAtom:
    name: str
    resname: str
    resnum: int
    icode: str
    chain: str
    x: float
    y: float
    z: float
    bfactor: float
    element: str = ""

    @property
    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain, self.resnum, self.name)

    @property
    def is_hydrogen(self) -> bool:
        """元素列优先; 缺失时退化为名称启发(H 开头或数字开头, 如 HZ2/1HB)。"""
        if self.element:
            return self.element.upper() == "H"
        n = self.name.strip()
        return bool(n) and (n[0] == "H" or n[0].isdigit())


def _kabsch_transform(src: "Sequence[Sequence[float]]",
                      dst: "Sequence[Sequence[float]]"):
    """最优刚体变换(含镜像拒绝): 把 src 叠到 dst, 返回 (R(3x3), t(3,))。

    用途(优化项 B1): 隐式溶剂 MD 后的去漂移刚体回正 —— 只旋转+平移, 不做镜像;
    若 SVD 行列式为负(数值/退化情形)先做一次翻转校正再重解。

    src/dst 为 (N,3) 可下标序列(支持 list of tuple 与 ndarray)。
    """
    import numpy as np
    P = np.asarray(src, dtype=float)
    Q = np.asarray(dst, dtype=float)
    if P.shape != Q.shape or P.shape[0] < 3:
        return np.eye(3), np.zeros(3)
    Pc, Qc = P - P.mean(axis=0), Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    if d < 0:                      # 拒绝镜像
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
    R = Vt.T @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def parse_pdb_atoms(pdb_path: str, records: Tuple[str, ...] = ("ATOM",)) -> List[PAtom]:
    atoms: List[PAtom] = []
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith(records):
                continue
            try:
                atoms.append(PAtom(
                    ln[12:16].strip(), ln[17:20].strip(), int(ln[22:26]),
                    ln[26].strip(), ln[21].strip(),
                    float(ln[30:38]), float(ln[38:46]), float(ln[46:54]),
                    float(ln[60:66]) if ln[60:66].strip() else 0.0,
                    ln[76:78].strip() if len(ln) >= 78 else ""))
            except (ValueError, IndexError):
                continue
    return atoms


def hetatm_resnames(pdb_path: str, keep_water: bool = False) -> List[str]:
    names: List[str] = []
    seen = set()
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("HETATM"):
                continue
            rn = ln[17:20].strip()
            if not keep_water and rn in ("HOH", "WAT", "DOD"):
                continue
            key = (rn, ln[21].strip(), ln[22:26].strip())
            if key not in seen:
                seen.add(key)
                names.append(rn)
    return sorted(set(names))


def kabsch_rmsd(p: List[Tuple[float, float, float]],
                q: List[Tuple[float, float, float]]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Kabsch 最优叠合。返回 (RMSD, 叠合后的 p 坐标)。p/q 等长。需要 numpy。"""
    import numpy as np
    P = np.asarray(p, dtype=float)
    Q = np.asarray(q, dtype=float)
    if P.shape != Q.shape or len(P) == 0:
        raise ValueError("kabsch_rmsd: 坐标集为空或长度不一致")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = math.copysign(1.0, float(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Pa = Pc @ R.T
    rmsd = float(np.sqrt(((Pa - Qc) ** 2).sum() / len(P)))
    return rmsd, [tuple(v) for v in (Pa + Q.mean(axis=0))]


def matched_rmsd(a: List[PAtom], b: List[PAtom],
                 atom_name: str = "CA", align: bool = False) -> Tuple[float, int]:
    """按 (chain,resnum,name) 匹配两批原子, 返回 (RMSD, 匹配数)。
    align=True 时先 Kabsch 叠合; 否则直接逐点 RMSD(适合'同坐标系未动'的对比)。"""
    bm = {x.key: x for x in b if (not atom_name or x.name == atom_name)}
    pairs = [(x, bm[x.key]) for x in a
             if (not atom_name or x.name == atom_name) and x.key in bm]
    if not pairs:
        return float("nan"), 0
    p = [x.coord for x, _ in pairs]
    q = [y.coord for _, y in pairs]
    if align:
        rmsd, _ = kabsch_rmsd(p, q)
        return rmsd, len(pairs)
    s = sum(math.dist(pc, qc) ** 2 for pc, qc in zip(p, q))
    return math.sqrt(s / len(pairs)), len(pairs)


def positions_to_angstrom(positions) -> List[Tuple[float, float, float]]:
    """把 OpenMM/PDBFixer 坐标序列统一换算成**埃**的纯 float 三元组。

    单位陷阱(2026-10-02 修复): `PDBFixer.positions` 由 openmm 维护, 其单位**恒为
    nm**; 而 `openmm.app.PDBFile.writeFile` 对无单位入参原样写出。原先只有
    `_minimize_local` 的回写路径(:527)做了 ×10, 直接 `writeFile(fixer.positions)`
    会把结构整体缩小 10 倍且不报错(→ `--no_minimize` / openmm 缺失的退化路径
    静默产出坐标 ÷10 的"坏但 ok"文件)。

    幂等设计: `_minimize_local` 回写后 positions 已是 float 元组(埃),
    再次调用本函数按元素类型识别后原样返回, 保证"转换只发生一次"。
    元素为 openmm Vec3(带 .x/.y/.z)→ nm → ×10; 元素为 tuple/list/float → 视为埃。
    """
    out: List[Tuple[float, float, float]] = []
    for p in positions:
        if isinstance(p, (tuple, list)):
            out.append((float(p[0]), float(p[1]), float(p[2])))
        else:  # openmm Vec3
            out.append((p.x * 10.0, p.y * 10.0, p.z * 10.0))
    return out


def parse_pdb_heavy_or_all(pdb_path: str):
    """解析 PDB 全部原子(含 H)。供单位/结构完整性自检用。"""
    return parse_pdb_atoms(pdb_path, records=("ATOM", "HETATM"))


def coordinate_span_A(pdb_path: str) -> Tuple[float, float, float]:
    """返回结构三方向坐标跨度 (sx, sy, sz), 单位**埃**。

    单位护栏用: nm/Å 混淆会让跨度差 10 倍, 一眼可辨(蛋白质口径 30~80 A)。
    """
    xs, ys, zs = [], [], []
    for a in parse_pdb_atoms(pdb_path):
        xs.append(a.x); ys.append(a.y); zs.append(a.z)
    if not xs:
        return (0.0, 0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def clash_check(pdb_path: str,
                focus_residues: Optional[Sequence[Tuple[str, int]]] = None,
                threshold_A: float = 2.2,
                exclude_seq_neighbors: int = 1) -> List[Dict[str, Any]]:
    """重原子 clash 检查(键连距离下限口径, 非 vdW 精确判定, 只作 QC 警告)。

    氢原子一律排除(带 H 的高分辨晶体中, H→重原子 1.7~2.2A 是键连/氢键距离,
    会计成假 clash)。focus_residues 给定((chain,resnum) 列表)时, 只报告至少
    一端落在焦点残基上的 clash 对; 同残基内部与相邻残基对跳过。
    """
    atoms = [a for a in parse_pdb_atoms(pdb_path) if not a.is_hydrogen]
    focus = set(focus_residues or [])
    out: List[Dict[str, Any]] = []
    for i in range(len(atoms)):
        a = atoms[i]
        a_focus = (a.chain, a.resnum) in focus if focus else True
        for j in range(i + 1, len(atoms)):
            b = atoms[j]
            same_res = (a.chain == b.chain and a.resnum == b.resnum)
            if same_res:
                continue
            if a.chain == b.chain and abs(a.resnum - b.resnum) <= exclude_seq_neighbors:
                continue
            b_focus = (b.chain, b.resnum) in focus if focus else True
            if focus and not (a_focus or b_focus):
                continue
            d = math.dist(a.coord, b.coord)
            if d < threshold_A:
                out.append({
                    "atom_a": f"{a.resname}{a.resnum}{a.chain}:{a.name}",
                    "atom_b": f"{b.resname}{b.resnum}{b.chain}:{b.name}",
                    "dist_A": round(d, 2)})
    out.sort(key=lambda c: c["dist_A"])
    return out


# ----------------------------------------------------------------------------
# 引擎抽象
# ----------------------------------------------------------------------------
class MutantEngine(ABC):
    """突变建模引擎统一接口。实现方保证:
    1. 不移动输入结构骨架(局部松弛仅限 restraint_radius 内);
    2. 输出 ATOM-only PDB, 保留原链 ID 与残基编号;
    3. mutate() 返回可 JSON 序列化的 dict(见 PdbfixerOpenMMEngine 实现)。
    """
    name: str = "abstract"
    desc: str = ""

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """(是否可用, 原因/版本信息)。"""

    @abstractmethod
    def mutate(self, receptor_pdb: str, mutations: Sequence[MutSpec],
               out_pdb: str, *, minimize: bool = True,
               restraint_radius: float = 8.0) -> Dict[str, Any]:
        """执行突变并写出 out_pdb, 返回报告 dict。"""


def _aa_present(fixer, chain_id: str) -> Dict[int, str]:
    """{resnum: resname3} (指定链)。"""
    out: Dict[int, str] = {}
    for ch in fixer.topology.chains():
        if ch.id != chain_id:
            continue
        for r in ch.residues():
            try:
                out[int(r.id)] = r.name
            except ValueError:
                continue
    return out


# 隐式溶剂局部 MD 的随机种子(优化项 B1)：不钉种子 → 初速随机 → 同输入两次跑出不同
# 坐标, 与"publication grade 可复现"冲突。默认固定, 可用环境变量 DOCK_RELAX_MD_SEED
# 覆盖(想做多次取样对比时换种子重跑)。
# 2026-10-04 全盘审计: 原 `int(os.environ.get(...))` 对非数字 env 在 import 期
# 裸栈(与第六轮 C7 env_float 统一口径同型), 改防御解析。
try:
    RELAX_MD_SEED: int = int(os.environ.get("DOCK_RELAX_MD_SEED", "20260401"))
except ValueError:
    RELAX_MD_SEED = 20260401

# 松弛模式(优化项 B1, 2026-10-04): none=只替换侧链不松弛, minimize=真空力场局部
# 最小化(历史默认), md=隐式溶剂(OBC/GBneck2)局部短程 MD —— 见 _relax_solvated_md
# docstring(显式 TIP3P 经实测对埋藏位点不可用, 已按实测结论改为隐式溶剂)。
RELAX_MODES: Tuple[str, ...] = ("none", "minimize", "md")


class PdbfixerOpenMMEngine(MutantEngine):
    """PDBFixer 换侧链 + 补缺失原子, OpenMM 局部最小化(真空或隐式溶剂, 可选)。"""
    name = "pdbfixer"
    desc = "PDBFixer(侧链替换) + OpenMM(局部受限最小化); 纯 Python, conda/pip 可装"

    def is_available(self) -> Tuple[bool, str]:
        try:
            import pdbfixer  # noqa: F401
        except ImportError:
            return False, "缺 pdbfixer: pip install pdbfixer (或 conda install -c conda-forge pdbfixer)"
        try:
            import openmm
            return True, f"pdbfixer + openmm {openmm.__version__}"
        except ImportError:
            return True, "pdbfixer 可用; 缺 openmm → 退化模式(不局部最小化)"

    # -- B 因子快照/回填 -------------------------------------------------
    @staticmethod
    def _bfact_snapshot(pdb_path: str) -> Tuple[Dict[Tuple[str, int, str], float],
                                                Dict[Tuple[str, int], float]]:
        per_atom: Dict[Tuple[str, int, str], List[float]] = {}
        per_res: Dict[Tuple[str, int], List[float]] = {}
        for a in parse_pdb_atoms(pdb_path):
            per_atom.setdefault(a.key, []).append(a.bfactor)
            per_res.setdefault((a.chain, a.resnum), []).append(a.bfactor)
        atom_b = {k: v[0] for k, v in per_atom.items()}
        res_b = {k: (sum(v) / len(v)) for k, v in per_res.items()}
        return atom_b, res_b

    @staticmethod
    def _strip_hydrogens(pdb_path: str) -> int:
        """删除 ATOM 行中的氢(输入晶体自带 H 时, 否则输出"部分残基带H/新侧链无H"
        的不一致文件; 加氢统一交给下游受体准备)。返回删除行数。"""
        kept, n = [], 0
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("ATOM"):
                    el = ln[76:78].strip() if len(ln) >= 78 else ""
                    nm = ln[12:16].strip()
                    if (el and el.upper() == "H") or \
                            (not el and nm and (nm[0] == "H" or nm[0].isdigit())):
                        n += 1
                        continue
                kept.append(ln)
        if n:
            with open(pdb_path, "w", encoding="utf-8") as f:
                f.writelines(kept)
        return n

    @staticmethod
    def _bfact_restore(pdb_path: str,
                       atom_b: Dict[Tuple[str, int, str], float],
                       res_b: Dict[Tuple[str, int], float]) -> int:
        """按 (chain,resnum,atom) 把 B 因子写回; 新原子取残基均值。返回回填行数。"""
        lines = []
        n = 0
        with open(pdb_path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("ATOM"):
                    try:
                        key = (ln[21].strip(), int(ln[22:26]), ln[12:16].strip())
                    except ValueError:
                        key = None
                    if key:
                        b = atom_b.get(key, res_b.get((key[0], key[1]), 0.0))
                        ln = ln[:60] + f"{b:6.2f}" + ln[66:]
                        n += 1
                lines.append(ln)
        with open(pdb_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return n

    # -- 主流程 ----------------------------------------------------------
    def mutate(self, receptor_pdb: str, mutations: Sequence[MutSpec],
               out_pdb: str, *, minimize: bool = True,
               restraint_radius: float = 8.0,
               relax: str = "", md_ps: float = 0.5) -> Dict[str, Any]:
        """建模并松弛, 结果写 out_pdb, 返回报告 dict。

        mutations 接受**任意数量**的组合突变(如 [F205G, S126A], 可同链也可跨链):
        按链分组后逐链调用 PDBFixer.applyMutations; WT 残基身份逐位点+逐链硬校验;
        relax=md 时各突变位点 Cα 共同作为局部松弛的放行中心。rep 新增
        n_mutations / is_combo / combo_chains 三个组合元信息字段。

        relax(优化项 B1, 2026-10-04): none/minimize/md, 见 RELAX_MODES。缺省行为
        完全等价于旧版(relax=="" 且 minimize=True → 真空局部最小化), 老调用方不受影响;
        relax="md" 走隐式溶剂(OBC)短程 MD(介电屏蔽下的极性侧链/盐桥/溶剂化效应,
        对埋藏位点同样成立 —— 显式 TIP3P 经实测不可用, 见 _relax_solvated_md)。
        """
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        mode = (relax or ("minimize" if minimize else "none")).strip().lower()
        if mode not in RELAX_MODES:
            raise ValueError(f"未知 relax 模式 {relax!r}(可选: {','.join(RELAX_MODES)})")

        rep: Dict[str, Any] = {
            "engine": self.name, "receptor_in": os.path.abspath(receptor_pdb),
            "out_pdb": os.path.abspath(out_pdb), "mutations": [],
            "minimized": False, "restraint_radius_A": restraint_radius,
            "ligands_dropped": [], "bfact_restored": False, "notes": [],
        }

        lig = hetatm_resnames(receptor_pdb)
        if lig:
            rep["ligands_dropped"] = lig
            LOG.warning(f"突变体输出为纯蛋白 PDB, 丢弃 HETATM: {','.join(lig)} "
                        f"(共晶配体在突变体上无确定放置; 盒子请用 --inherit_box 继承 WT)")

        atom_b, res_b = self._bfact_snapshot(receptor_pdb)

        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        # 只做定点突变, 不补建模晶体缺失残基/loop(那会引入预测坐标、污染对比)
        n_missing_res = sum(len(v) for v in fixer.missingResidues.values())
        if n_missing_res:
            rep["notes"].append(
                f"晶体缺失残基 {n_missing_res} 个已跳过(不补建模, 保持与 WT 可比)")
        fixer.missingResidues = {}
        fixer.removeHeterogens(keepWater=False)

        # 目标链: 取首个突变指定的链作为 primary(空 chain 补全用), 否则唯一链/第一条链
        chains = [c.id for c in fixer.topology.chains()]
        muts = list(mutations)
        if not muts:
            raise ValueError("未提供任何突变")
        primary = next((m.chain for m in muts if m.chain), chains[0] if chains else "")
        if not primary:
            raise ValueError("受体无链; 突变也未用 ':链' 显式指定链")
        if primary not in chains:
            raise ValueError(f"链 {primary} 不存在于受体(现有链: {chains})")
        muts = [MutSpec(m.wt3, m.resnum, m.mut3, m.chain or primary) for m in muts]
        # 按链分组: PDBFixer 的 applyMutations 一次只接一条链 → 每条链分别调用
        # (组合突变可跨链, 如催化三联体多残基塑形; 同链多突变自然落在同一组)
        by_chain: Dict[str, List[MutSpec]] = {}
        for m in muts:
            by_chain.setdefault(m.chain, []).append(m)

        # WT 残基身份核对(逐位点、逐链; 防止编号错位静默错突)
        for m in muts:
            present = _aa_present(fixer, m.chain)
            got = present.get(m.resnum)
            if got is None:
                raise ValueError(f"链 {m.chain} 无残基 {m.resnum} ({m.label})")
            if got != m.wt3:
                raise ValueError(f"{m.label} 与结构不符: 链 {m.chain} 第 {m.resnum} 位 "
                                 f"实为 {got}({AA3_TO_1.get(got, '?')}), 非 {m.wt3}")

        for ch, ch_muts in by_chain.items():
            fixer.applyMutations([m.pdbfixer for m in ch_muts], ch)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        for m in muts:
            rep["mutations"].append({"label": m.label, "chain": m.chain,
                                     "resnum": m.resnum, "wt": m.wt3,
                                     "mut": m.mut3, "applied": True})
            if m.wt3 == m.mut3:
                # 同残基同氨基酸"突变"(如 Phe→Phe)是**刻意的对称对照**: 序列不变, 但会照走
                # applyMutations+重建侧链+局部最小化, 从而得到"与真实突变体同一松弛
                # 路径、位点仍为 WT 残基"的参照 —— 用来把"参照结构没松弛"那部分偏置
                # (实测 +9.4%)从突变效应里剥离。必须显式记一笔, 否则产物里的
                # {"wt": "PHE", "mut": "PHE"} 会被当成一个真突变体来读。
                rep["notes"].append(
                    f"{m.label} 是同残基'突变'(wt==mut): 序列不变, 仅作与突变体同松弛"
                    f"路径的**对称对照**(用于剥离\"参照未松弛\"偏置); 不得当作突变体描述")

        # 组合突变元信息(2026-10-07, #280): 多突变组合的口径汇总
        rep["n_mutations"] = len(rep["mutations"])
        rep["is_combo"] = len(rep["mutations"]) > 1
        if rep["is_combo"]:
            rep["combo_chains"] = sorted({m["chain"] for m in rep["mutations"]})

        rep["relax_mode"] = mode
        if mode == "md":
            rep["minimized"] = self._relax_solvated_md(fixer, muts, primary,
                                                      restraint_radius, rep,
                                                      md_ps=float(md_ps))
        elif mode == "minimize" and minimize:
            rep["minimized"] = self._minimize_local(fixer, muts, primary,
                                                    restraint_radius, rep)
        else:
            rep["minimized"] = False
            rep["notes"].append("未做局部松弛(--relax none / --no_minimize)")

        os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
        # 单位**: 必须显式换算为埃**。fixer.positions 是 nm, writeFile 对无单位入参
        # 原样写出 —— 不做换算则产出坐标 ÷10 的坏文件且无任何报错。
        # (幂等, `_minimize_local` 已换成埃也只会原样返回)
        pos_A = positions_to_angstrom(fixer.positions)
        rep["positions_unit"] = "angstrom"
        with open(out_pdb, "w", encoding="utf-8") as f:
            # keepIds=True: 保留原链 ID 与残基编号(默认 False 会顺序重排, 编号错位)
            PDBFile.writeFile(fixer.topology, pos_A, f, keepIds=True)
        n_h = self._strip_hydrogens(out_pdb)
        if n_h:
            rep["notes"].append(f"已剔除输入晶体自带氢原子 {n_h} 个(加氢由下游统一处理)")
        rep["bfact_restored"] = bool(self._bfact_restore(out_pdb, atom_b, res_b))
        return rep

    # -- OpenMM 局部受限最小化 -------------------------------------------
    @staticmethod
    def _minimize_local(fixer, muts: Sequence[MutSpec], chain_id: str,
                        radius: float, rep: Dict[str, Any]) -> bool:
        """真空力场局部最小化。

        推理: amber14 模板匹配要求残基带氢 → 在临时 Modeller 副本上加氢最小化,
             结束后只把**重原子**坐标回写 fixer(氢丢弃, 输出保持 ATOM-only,
             加氢/质子化仍由 DockBuddy 受体准备链路统一负责)。"""
        try:
            import openmm as mm
            from openmm import unit
        except ImportError:
            LOG.warning("openmm 不可用, 跳过局部最小化(仅 PDBFixer 侧链替换)")
            rep["notes"].append("openmm 缺失: 未局部最小化")
            return False

        # 突变残基 CA 坐标为放行中心(用 nm)
        centers: List[Tuple[float, float, float]] = []
        for a in fixer.topology.atoms():
            r = a.residue
            if a.name == "CA" and \
                    any(r.chain.id == m.chain and int(r.id) == m.resnum for m in muts):
                p = fixer.positions[a.index]
                centers.append((p.x, p.y, p.z))

        ff = mm.app.ForceField("amber14-all.xml")
        modeller = mm.app.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(ff, pH=7.0)
        system = ff.createSystem(modeller.topology, constraints=mm.app.HBonds)
        restr = mm.CustomExternalForce(
            "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restr.addPerParticleParameter("k")
        restr.addPerParticleParameter("x0")
        restr.addPerParticleParameter("y0")
        restr.addPerParticleParameter("z0")
        k_rest = 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2
        r_nm = radius / 10.0
        m_atoms = list(modeller.topology.atoms())
        m_pos = modeller.positions
        n_free = 0
        for a in m_atoms:
            p = m_pos[a.index]
            is_h = a.element is not None and a.element.symbol == "H"
            free = (not is_h) and any(
                math.dist((p.x, p.y, p.z), c) <= r_nm for c in centers)
            n_free += 1 if free else 0
            restr.addParticle(a.index,
                              (0.0 if free else k_rest, p.x, p.y, p.z))
        system.addForce(restr)
        integ = mm.LangevinIntegrator(300 * unit.kelvin,
                                      1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        # 显式 CPU 平台(可经 DOCK_OPENMM_PLATFORM=CUDA 切 GPU): 缺省构造会让 OpenMM
        # 自动探测 OpenCL/CUDA 平台, 在 GPU 驱动不完整/冲突的环境(如 PyCharm 运行配置的
        # PATH 与终端不同)会触发 Windows fast-fail 原生崩溃(exit 0xC06D007F)。局部最小化
        # 规模小, CPU 足够; 想提速可显式切 CUDA。
        try:
            from dockbuddy_common import openmm_platform
            platform = openmm_platform()
            ctx = mm.Context(system, integ, platform)
        except Exception:
            ctx = mm.Context(system, integ)
        ctx.setPositions(modeller.positions)
        mm.LocalEnergyMinimizer.minimize(
            ctx, tolerance=50.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=300)
        state = ctx.getState(getPositions=True)
        new_pos = state.getPositions()

        # 重原子坐标回写 fixer.positions(按 chain/resid/atom name 匹配)
        new_map: Dict[Tuple[str, str, str], Any] = {}
        for a in m_atoms:
            if a.element is not None and a.element.symbol == "H":
                continue
            r = a.residue
            new_map[(r.chain.id, r.id, a.name)] = new_pos[a.index]
        fixer_pos = list(fixer.positions)
        n_back = 0
        for a in fixer.topology.atoms():
            r = a.residue
            key = (r.chain.id, r.id, a.name)
            if key in new_map:
                fixer_pos[a.index] = new_map[key]
                n_back += 1
        # 纯 float 元组列表(单位: 埃)。writeFile 对无单位输入不做换算原样写出,
        # 而 openmm 内部坐标是 nm → 必须显式 ×10 换成埃, 否则结构整体缩 10 倍
        fixer.positions = [(p.x * 10.0, p.y * 10.0, p.z * 10.0) for p in fixer_pos]
        rep["n_free_atoms"] = n_free
        rep["n_restrained_atoms"] = len(m_atoms) - n_free
        rep["notes"].append(f"局部最小化回写重原子 {n_back} 个(临时加氢已丢弃)")
        return True

    # -- 隐式溶剂(OBC)局部短程 MD 松弛(优化项 B1) -------------------------
    @staticmethod
    def _relax_solvated_md(fixer, muts: Sequence[MutSpec], chain_id: str,
                           radius: float, rep: Dict[str, Any],
                           md_ps: float = 0.5, temp_K: float = 300.0) -> bool:
        """局部**隐式溶剂**短程 MD 松弛(优化项 B1, 2026-10-04)。

        与 _minimize_local(真空 amber14 + 局部谐振约束 + 单步最小化)的差别, 以及
        为什么需要它: 真空里静电不被介电屏蔽, 盐桥/极性侧链(HIS 互变异构、D/E 与
        K/R 盐桥)的梯度被放大约 80 倍, 侧链可以靠"自己把自己推开"骗过梯度 —— 而
        突变改造最在意的恰恰是极性效应。这里用 OBC(GBneck2)隐式溶剂, 溶剂介电常数
        78.5, 无需显式水分子就给出正确的屏蔽与溶剂化项。

        为什么**不是**显式 TIP3P(实测结论, 别把它"优化"回去): 实测某深埋芳香
        锚定残基体系(突变位点深埋在芳香笼里)—— 以该残基 Cα 为心: 水盒 L<=2.8 nm
        时 addSolvent 加水 0 个(静默退化成真空),
        L=3.6 nm 有 245 个水但最近的那个也距 Cα 11.5 A。即位点本身没有可达的溶剂化
        层, 铺显式水要么 0 个、要么只能在"够不着的体相"里加水, 物理上是假的。隐式溶剂
        对埋藏位点和表面位点同样成立, 故 B1 统一走隐式溶剂。

        流程: 加氢 → amber14 + OBC 隐式溶剂体系 → 半径外原子强谐振约束(k=1000) →
        最小化(50) → Langevin MD(md_ps, 300 K, 2 fs) → 再最小化(30) → 重原子回写
        fixer → **刚体回正**(Kabsch 消掉 MD 带来的整体漂移, 保证突变体与 WT 可直接比对)。

        刚体回正为什么必须有: 约束只钉住半径外原子, 允许整体微小平移/转动; 而
        DockBuddy 的突变体对比依赖"与 WT 注册一致"(同一盒子、同一种子矩阵、
        Cα RMSD 可直接比)。漂移不消掉会污染这一整套比较。

        成本(实测, 本机 OpenMM CPU 平台 / 3664 粒子): 加氢 ~15 s、minimize(50)
        ~40 s、MD(0.5 ps) ~30 s、minimize(30) ~25 s —— 合计约 2 分钟量级。OpenMM 的
        "CPU" 平台是单线程保守实现, 快机会明显更快; 想改用 GPU 需自行评估驱动冲突
        (本项目因此长期钉死 CPU, 见 _minimize_local 的注释)。md_ps=0 可只做最小化。

        确定性: Langevin 初速来自随机源, 不钉种子就无法逐位复现 —— 对"publication
        grade"的对比是硬伤。故默认走固定种子 RELAX_MD_SEED(可用环境变量覆盖),
        同一输入+同一平台必然给出同一组坐标。
        """
        try:
            import openmm as mm
        except ImportError:
            LOG.warning("openmm 不可用, 跳过隐式溶剂松弛(仅 PDBFixer 侧链替换)")
            rep["notes"].append("openmm 缺失: 未做隐式溶剂松弛")
            return False
        from openmm import unit

        # 放行中心 = 突变残基 Cα(nm)
        centers: List[Tuple[float, float, float]] = []
        site_ca_nm: Tuple[float, float, float] = ()
        for a in fixer.topology.atoms():
            r = a.residue
            if a.name == "CA" and \
                    any(r.chain.id == m.chain and int(r.id) == m.resnum for m in muts):
                p = fixer.positions[a.index]
                centers.append((p.x, p.y, p.z))
                if not site_ca_nm:
                    site_ca_nm = (p.x, p.y, p.z)
        if not centers:
            LOG.warning("未定位到突变残基 Cα, 隐式溶剂 MD 跳过")
            return False

        ff = mm.app.ForceField("amber14-all.xml")
        modeller = mm.app.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(ff, pH=7.0)
        system = ff.createSystem(
            modeller.topology, nonbondedMethod=mm.app.NoCutoff,
            constraints=mm.app.HBonds)
        # 显式溶剂对埋藏位点不可用(实测, 见 docstring), 改用 OBC 隐式溶剂。本版
        # OpenMM 的 app 层 createSystem 已不再接受 implicitSolvent= 参数(8.6 起
        # 报 "was specified but never used"), 必须手工装配 GBSAOBCForce。
        # 装配参数(半径换算 0.5612… / screen 0.5)集中放在 libdock.solvents, 与
        # libdock.rescore(MM-GBSA 重评分)共用同一份口径, 避免两处漂移。
        try:
            from dockbuddy_mutate.solvents import (attach_obc_force as _attach_obc,
                                          OBC_ID as _OBC_ID)
            if _attach_obc(system, LOG) is None:
                raise RuntimeError("attach_obc_force 返回 None")
            rep["solvent_model"] = _OBC_ID
        except Exception as e:                              # pragma: no cover
            LOG.warning("GB 隐式溶剂不可用(%s), 退回真空局部最小化", e)
            rep["notes"].append(f"OBC 隐式溶剂装配失败, 已退回局部最小化: {e}")
            return PdbfixerOpenMMEngine._minimize_local(
                fixer, muts, chain_id, radius, rep)

        # 位点埋深(只读诊断, 不影响坐标): 5 A 内的蛋白重原子数 —— 大 = 结构核心,
        # 侧面印证"显式溶剂进不去、只能靠隐式溶剂"的结论。
        try:
            import numpy as _npb
            _ca = _npb.array(site_ca_nm, dtype=float)
            _all = _npb.array(
                [(p.x, p.y, p.z) for p in fixer.positions], dtype=float) * 10.0
            rep["site_neighbors_5A"] = int(
                (_npb.linalg.norm(_all - _ca * 10.0, axis=1) <= 5.0).sum())
        except Exception:
            pass

        k_rest = 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2
        r_nm = radius / 10.0
        restr = mm.CustomExternalForce(
            "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restr.addPerParticleParameter("k")
        restr.addPerParticleParameter("x0")
        restr.addPerParticleParameter("y0")
        restr.addPerParticleParameter("z0")
        m_atoms = list(modeller.topology.atoms())
        m_pos = modeller.positions.value_in_unit(unit.nanometer)
        n_free = 0
        for a in m_atoms:
            p = m_pos[a.index]
            is_h = a.element is not None and a.element.symbol == "H"
            free = (not is_h) and any(
                float(math.dist((p[0], p[1], p[2]), c)) <= r_nm for c in centers)
            n_free += 1 if free else 0
            restr.addParticle(a.index, (0.0 if free else k_rest,
                                        float(p[0]), float(p[1]), float(p[2])))
        system.addForce(restr)

        integ = mm.LangevinIntegrator(temp_K * unit.kelvin,
                                      1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        try:                       # 固定随机种子 → MD 可逐位复现(见 docstring)
            integ.setRandomNumberSeed(int(RELAX_MD_SEED))
        except Exception:          # pragma: no cover
            pass
        # 显式 CPU 平台(同 _minimize_local 的理由; 可经 DOCK_OPENMM_PLATFORM=CUDA 切 GPU):
        # 缺省探测会在 GPU 驱动冲突的环境原生 fast-fail(Windows 0xC06D007F)。
        try:
            from dockbuddy_common import openmm_platform
            ctx = mm.Context(system, integ, openmm_platform())
        except Exception:
            ctx = mm.Context(system, integ)
        ctx.setPositions(modeller.positions)
        tol = 10.0 * unit.kilojoule_per_mole / unit.nanometer
        # 迭代数是成本主要来源(实测本机 minimize(1 次)≈0.8 s), 取小即可:
        # 松弛只作用于局部, 松弛完还要做 MD, 过度最小化没有额外收益。
        mm.LocalEnergyMinimizer.minimize(ctx, tolerance=tol, maxIterations=50)
        if md_ps > 0:
            ctx.setVelocitiesToTemperature(temp_K * unit.kelvin)
            # 积分步数走 integrator(新版 OpenMM 已移除 ctx.step), 用 getattr 兼容。
            n_steps = max(1, int(round(md_ps * 1000.0 / 2.0)))   # 2 fs/步
            stepper = getattr(ctx, "step", None)
            if stepper is not None:
                stepper(n_steps)
            else:
                integ.step(n_steps)
        mm.LocalEnergyMinimizer.minimize(ctx, tolerance=tol, maxIterations=30)

        rep["n_free_atoms"] = int(n_free)
        rep["n_restrained_atoms"] = int(restr.getNumParticles()) - int(n_free)
        rep["md_ps"] = round(float(md_ps), 2)
        rep["md_temperature_K"] = float(temp_K)
        rep["md_seed"] = int(RELAX_MD_SEED)
        rep["solvent_model"] = rep.get(
            "solvent_model", "OBC, solvent_dielectric=78.5")
        new_pos = ctx.getState(getPositions=True).getPositions()

        # --- 去氢, 只留蛋白重原子坐标; 与 fixer 按 (chain,resid,name) 对齐 ---
        # 注意不能用 a.index: modeller 删过氢之后索引与 fixer.topology 不同构。
        keys, keep_rows, fixer_rows = set(), [], []
        for i, a in enumerate(m_atoms):
            if a.element is not None and a.element.symbol == "H":
                continue
            r = a.residue
            key = (r.chain.id, r.id, a.name)
            keys.add(key)
            keep_rows.append((new_pos[i].x, new_pos[i].y, new_pos[i].z))
        fixer_atoms = list(fixer.topology.atoms())
        for a in fixer_atoms:
            r = a.residue
            if (r.chain.id, r.id, a.name) not in keys:
                continue
            p = fixer.positions[a.index]
            fixer_rows.append((p.x, p.y, p.z))
        if len(keep_rows) != len(fixer_rows):
            LOG.warning("MD 后蛋白重原子数与 fixer 不匹配 "
                        f"({len(keep_rows)} vs {len(fixer_rows)}), 放弃刚体回正以保持对齐")
            return False
        # 刚体回正: 源 = MD 后蛋白坐标, 靶 = fixer 原坐标, 求最优刚体变换后整体系施加
        import numpy as _np
        rot, tran = _kabsch_transform(keep_rows, fixer_rows)
        allpos = _np.array([[p.x, p.y, p.z] for p in new_pos])
        fixed_all = allpos @ rot.T + tran
        keep_key_idx = {}
        for i, a in enumerate(m_atoms):
            if a.element is not None and a.element.symbol == "H":
                continue
            r = a.residue
            keep_key_idx[(r.chain.id, r.id, a.name)] = i
        fixer_pos = list(fixer.positions)
        n_back = 0
        for a in fixer_atoms:
            r = a.residue
            key = (r.chain.id, r.id, a.name)
            i = keep_key_idx.get(key)
            if i is None:
                continue
            fixer_pos[a.index] = tuple(float(x) for x in fixed_all[i])  # nm
            n_back += 1
        # 坐标一律写成**纯 float** 元组: fixer.positions 原元素是 Quantity(有 .x),
        # 而上面已混入裸元组, 混合后取 .x 会 AttributeError —— 统一归一化再赋回。
        def _xyzz(v):
            if hasattr(v, "x"):
                return (v.x * 10.0, v.y * 10.0, v.z * 10.0)
            return (float(v[0]) * 10.0, float(v[1]) * 10.0,
                    float(v[2]) * 10.0)
        fixer.positions = [_xyzz(p) for p in fixer_pos]
        rep["recentered"] = True
        rep["md_completed"] = True
        rep["notes"].append(
            f"隐式溶剂局部 MD({md_ps} ps, {temp_K} K)完成, "
            f"已刚体回正并回写重原子 {n_back} 个")
        return True



# ----------------------------------------------------------------------------
# 预留引擎(接口桩): 检测环境并给出明确指引, 将来按同一接口补实现
# ----------------------------------------------------------------------------
class _StubEngine(MutantEngine):
    binary_env: str = ""
    binary_names: Tuple[str, ...] = ()
    install_hint: str = ""

    def _find_binary(self) -> str:
        if self.binary_env and os.environ.get(self.binary_env):
            p = os.environ[self.binary_env]
            if os.path.exists(p):
                return p
        for n in self.binary_names:
            p = shutil.which(n)
            if p:
                return p
        return ""

    def is_available(self) -> Tuple[bool, str]:
        p = self._find_binary()
        if p:
            return False, f"已检测到 {p}, 但该引擎实现尚未接入(接口预留)"
        return False, f"未安装; {self.install_hint}(接口预留, 未实现)"

    def mutate(self, receptor_pdb, mutations, out_pdb, *, minimize=True,
               restraint_radius=8.0):
        raise NotImplementedError(
            f"引擎 {self.name} 为预留接口, 尚未实现; 请用 --engine pdbfixer, "
            f"或按 MutantEngine 接口在 libmutate.py 中补充实现")


class FoldXEngine(_StubEngine):
    name = "foldx"
    desc = "FoldX BuildModel(侧链替换+邻域重排, 附 ΔΔG); 需学术注册下载二进制"
    binary_env = "FOLDX_BINARY"
    binary_names = ("foldx", "foldx.exe", "FoldX")
    install_hint = "https://foldxsuite.crg.eu 注册后下载, 设环境变量 FOLDX_BINARY"


class SCWRL4Engine(_StubEngine):
    name = "scwrl4"
    desc = "SCWRL4(backbone-dependent rotamer 侧链替换); 学术免费二进制"
    binary_env = "SCWRL4_BINARY"
    binary_names = ("scwrl4", "Scwrl4", "Scwrl4.exe")
    install_hint = "http://dunbrack.fccc.edu/scwrl4 申请后安装, 设 SCWRL4_BINARY"


class ModellerEngine(_StubEngine):
    name = "modeller"
    desc = "MODELLER mutate_model(侧链替换+CG/短MD局部精修); 需 license key"
    binary_env = ""
    binary_names = ()
    install_hint = "https://salilab.org/modeller 注册获取 license 后 pip/conda 安装"


class PyMOLEngine(_StubEngine):
    name = "pymol"
    desc = "PyMOL Mutagenesis wizard(仅侧链替换, 不重排邻居); 适合演示不宜进流水线"
    binary_env = ""
    binary_names = ()
    install_hint = "安装 PyMOL (schrodinger 或开源版)"


ENGINES: Dict[str, type] = {
    "pdbfixer": PdbfixerOpenMMEngine,
    "foldx": FoldXEngine,
    "scwrl4": SCWRL4Engine,
    "modeller": ModellerEngine,
    "pymol": PyMOLEngine,
}
DEFAULT_ENGINE = os.environ.get("DOCK_MUT_ENGINE", "pdbfixer")


def get_engine(name: str = "") -> MutantEngine:
    key = (name or DEFAULT_ENGINE).strip().lower()
    if key not in ENGINES:
        raise ValueError(f"未知引擎 '{key}'; 可选: {','.join(sorted(ENGINES))}")
    return ENGINES[key]()


def list_engines() -> List[Dict[str, Any]]:
    out = []
    for name, cls in sorted(ENGINES.items()):
        e = cls()
        ok, why = e.is_available()
        out.append({"name": name, "available": ok, "detail": why, "desc": e.desc})
    return out
