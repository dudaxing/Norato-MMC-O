# GPTO 论文方法与案例的 Python 复现

本项目用现代 NumPy/SciPy 重新实现 Smith 与 Norato 的 geometry
projection topology optimization（GPTO）方法，并复现论文中的 MBB 梁、
L 型支架和三维悬臂梁案例。实现以论文案例定义和作者 MATLAB 代码为基准；
后续 PyGPTO 只作为交叉验证来源，因为其案例参数和当前运行环境都与论文存在偏差。

论文与上游固定版本：

- [Smith & Norato (2020), DOI](https://doi.org/10.1007/s00158-020-02552-0)
- [GPTO/MATLAB `ead7250`](https://github.com/jnorato/GPTO/tree/ead7250e007d4c185de59f8eee6b88a35be39550)
- [PyGPTO `8157ca2`](https://github.com/jnorato/PyGPTO/tree/8157ca2f5b82feb4a2032fd4501c9f49a58b7e8c)

## 复现口径

“复现”分成三个可验证层级：

1. 公式与灵敏度：投影、聚合、Q4/H8 有限元和解析梯度均有单元/有限差分测试。
2. 静态状态：给定相同设计，Python 与 MATLAB 的柔度和体积分数达到浮点误差量级一致。
3. 优化轨迹：MMA 对渐近线实现、barrier 解法和 MATLAB 版本敏感，因此报告结果差异，
   不把相近终态误称为逐步完全一致。

当前验证结果：

| 案例/状态 | Python | MATLAB/论文基准 | 结论 |
|---|---:|---:|---|
| MBB 初始柔度 | 45.526963475 | 45.526963456 | 静态吻合 |
| MBB 本实现终态 | 4.256798501 / 88 次 | 4.201067 / 88 次（论文） | 柔度相差 1.33% |
| L 型初始柔度 | 52.333199330 | 52.333199330 | 全精度网格下吻合 |
| L 官方终态柔度 | 2.846071785 | 2.846071785 | 静态吻合 |
| L 本实现终态 | 2.846612664 / 83 次 | 2.846072 / 64 次（论文） | 柔度相差 0.019% |
| 3D 缩小案例柔度 | 28.117944768 | 28.117944768 | 静态吻合 |
| 3D 官方存档柔度 | 1.333815329 | 1.333815330 | 相对误差 (8.9\times10^{-10}) |

完整数字和解释见 [复现报告](docs/REPRODUCTION_REPORT.md)，机器可读快照见
[verified_results.json](validation/verified_results.json)。

## 安装

需要 Python 3.11 或更新版本。Windows PowerShell 示例：

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[test,visualization]"
```

也可不安装，临时设置 `PYTHONPATH=src` 后使用 `py -3 -m gpto`。

## 快速验证

```powershell
# 22 项公式、梯度、有限元和跨语言回归测试
python -m pytest -q

# 查看案例
gpto cases

# 快速全链路分析和有限差分检查
gpto analyze mbb2d --profile tiny
gpto gradient-check mbb2d --profile smoke --indices 0,1,128

# 论文网格初始状态与官方存档状态
gpto analyze mbb2d --profile paper --design initial
gpto analyze lbracket2d --profile paper --design reference --output results/lbracket2d/reference-paper

# 完整二维优化
gpto optimize mbb2d --profile paper --output results/mbb2d/paper-run
gpto optimize lbracket2d --profile paper --output results/lbracket2d/paper-run
```

三维论文网格包含 128,000 个 H8 单元和 408,483 个自由度。实测纯 CPU 稀疏
装配和 Jacobi-CG 对官方存档设计耗时约 3.20 分钟、峰值工作集约 1.90 GB，
应先执行：

```powershell
gpto analyze cantilever3d --profile tiny
gpto analyze cantilever3d --profile paper --design reference --output results/cantilever3d/reference-paper
```

`paper` 使用论文网格，`smoke` 保留合理投影窗口但降低网格规模，`tiny` 仅用于
快速检查程序链路；缩小网格结果不能和论文数值直接比较。

## 输出

默认输出到 `results/<case>/...`，包括：

- `*.json`：参数、网格、柔度、体积分数、求解残差和论文基准；
- `*.npz`：设计变量、显式几何、密度、位移和解析梯度；
- `*.vtk`：可由 ParaView 打开的单元体积/刚度密度；
- `*.png`：二维显式杆件与投影密度、三维杆件示意；
- `history.csv`、`history.png`：优化历史。

已有 NPZ/JSON 可用 `tools/render_saved_evaluation.py` 重新绘图，不会再次执行
昂贵的有限元求解。

### 论文 Fig. 4--18 完整图集

在已生成上述全分辨率结果后，可一键重绘论文的 15 幅案例图：

```powershell
gpto figures
```

默认产物包括：

- `output/figures/gpto-paper-results/fig04_*.png` 到 `fig18_*.png`；
- `contact_sheet.png` 总览和可点击的 `index.html` 图廊；
- `manifest.json`，逐图记录数据来源、数值状态和保真限制；
- `output/pdf/gpto-paper-results-atlas.pdf`，每图一页的 PDF 图册。

这里严格沿用上游绘图语义：显式杆透明度为 `alpha²`；MBB 图 7 使用
惩罚后的有效密度；L 支架图 12 与三维图 18 使用未惩罚组合密度；图 14
对全部 66 个缩放变量使用步长 `1e-6` 的前向差分。图 18 复现
Cell Data to Point Data 后的 `rho=0.5` 等值面，但论文没有发布 ParaView
相机/光照状态，因此其材质和照明属于视觉近似。

## 实现范围

- 有限线段/胶囊构件的二维圆冠和三维球冠投影；
- SIMP、RAMP；论文 modified p-norm，以及 p-mean、KS 和概率并集实验模式；
- Q4 平面应力和 H8 三维线弹性，直接求解或 Jacobi-PCG；
- 共享端点设计变量、缩放、解析 compliance/volume 灵敏度；
- 独立实现的单体积约束 MMA 特例；
- 论文三例、官方存档设计、机器可读结果和 VTK 输出。

公式、代码对应关系和论文排版勘误见 [方法说明](docs/METHOD.md)。

## 许可证

本复现采用 CC BY-NC 4.0，详见 [LICENSE.md](LICENSE.md)。本项目没有复制
PyGPTO 中 GPL 许可的 `MMA.py`；`src/gpto/optimizer.py` 是针对三例单约束问题的
独立实现。上游代码和论文各自的许可与署名仍然适用。
