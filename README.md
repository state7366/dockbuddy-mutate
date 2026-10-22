# dockbuddy-mutate

蛋白定点突变建模，从 DockBuddy `pipeline/libmutate.py` 抽出的解耦版本。

给定 WT 结构 + 突变，产出突变体 PDB（PDBFixer 侧链替换 + OpenMM 局部松弛，带 QC：Cα RMSD/clash/坐标跨度）。

## 安装

```bash
pip install dockbuddy-mutate \
  --index-url https://pypi.cnb.cool/vcinex/bioinfo/-/packages/simple
# 实际建模需 OpenMM/PDBFixer
pip install dockbuddy-mutate[openmm]
```

## 使用

```python
from dockbuddy_mutate import parse_mutations, get_engine, parse_pdb_atoms, matched_rmsd

muts = parse_mutations(["F205G"], default_chain="A")
eng = get_engine("pdbfixer")
rep = eng.mutate("wt.pdb", muts, "mutant.pdb")   # 侧链替换 + 松弛
```

## API

| 函数/类 | 说明 |
|---|---|
| `parse_mutation` / `parse_mutations` | 解析突变（F205G / F205G,S126A） |
| `parse_pdb_atoms` / `matched_rmsd` | PDB 解析 / Cα RMSD |
| `clash_check` / `coordinate_span_A` | 结构 QC |
| `MutantEngine` / `PdbfixerOpenMMEngine` | 突变引擎（PDBFixer + OpenMM） |
| `get_engine` / `list_engines` | 引擎选择 |

硬依赖 `dockbuddy-common`；OpenMM/PDBFixer 懒加载（建模时才需要）。
