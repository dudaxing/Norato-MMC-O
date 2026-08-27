# 复现报告

## 证据与固定版本

本次复现使用三类一手证据：

- [论文 DOI](https://doi.org/10.1007/s00158-020-02552-0) 与
  [作者开放 PDF](https://sol.engr.uconn.edu/wp-content/uploads/sites/918/2020/03/Smith_et_al-2020-Structural_and_Multidisciplinary_Optimization.pdf)；
- [MATLAB GPTO 固定提交](https://github.com/jnorato/GPTO/tree/ead7250e007d4c185de59f8eee6b88a35be39550)；
- [PyGPTO 固定提交](https://github.com/jnorato/PyGPTO/tree/8157ca2f5b82feb4a2032fd4501c9f49a58b7e8c)。

包内二进制网格和参考几何的来源记录在 `src/gpto/data/sources.json`。
L 型支架使用 MATLAB 原始 `.m` 的全精度坐标；`tools/convert_matlab_mesh.py`
可重新生成它，避免 PyGPTO 四位小数截断造成的初始柔度偏差。

## 案例定义

| 项目 | MBB 半梁 | L 型支架 | 3D 悬臂 |
|---|---|---|---|
| 设计域 | (20\times5) | (100\times100) 去右上 (60\times60) | (20\times10\times10) |
| 论文网格 | 200×50 Q4 | 6,123 Q4 / 6,320 节点 | 80×40×40 H8 |
| 初始构件 | 32 根浮动杆 | 12 点共享的 21 根杆 | 16 根浮动短杆 |
| 初始半径 | 0.25 | 2 | 0.75 |
| 半径界 | [0.2499, 0.2501] | [2, 3] | [0.5, 1] |
| 体积上限 | 0.45 | 0.30 | 0.10 |
| 线性求解 | 稀疏直接 | 稀疏直接 | Jacobi-PCG |
| 论文结果 | 88 次，C=4.201067 | 64 次，C=2.846072 | 106 次，63 min |

3D 论文没有报告标量柔度；这里只比较它明确给出的迭代数/时间，并把上游
`.mat` 的静态状态作为代码级 golden，不能倒推成论文图 15–18 的公开数值。

## 静态状态验证

### 初始设计

| 案例 | 指标 | 本实现 | MATLAB golden | 差异 |
|---|---|---:|---:|---:|
| MBB | compliance | 45.5269634755 | 45.5269634564 | (4.2\times10^{-10}) relative |
| MBB | volume fraction | 0.33303274417484 | 0.33303274417483 | 舍入量级 |
| L 型 | compliance | 52.3331993301 | 52.3331993303 | (4.2\times10^{-12}) relative |
| L 型 | volume fraction | 0.170167616957038 | 同左 | 舍入量级 |
| 3D 缩小 20×10×10 | compliance | 28.1179447679 | 28.1179447679 | (<10^{-12}) relative |
| 3D 缩小 20×10×10 | volume fraction | 0.02273342433836 | 同左 | 舍入量级 |

MBB 的约 (2\times10^{-8}) 柔度绝对差来自 SciPy/MATLAB 稀疏线性代数的浮点
顺序，并非投影或有限元公式差异。

### 官方存档设计

| 状态 | Python compliance | MATLAB compliance | Python vf | MATLAB vf |
|---|---:|---:|---:|---:|
| MBB `.mat` | 4.3099304326 | 4.3099304319 | 0.449876064159 | 0.449876064159 |
| L connected `.mat` | 2.8460717850 | 2.8460717850 | 0.300000636757 | 0.300000636757 |
| 3D `.mat` | 1.3338153285 | 1.3338153297 | 0.099991061019 | 0.099991061019 |

MBB 存档并不是论文 Fig. 6/7 的最终状态，因为其柔度 4.30993 与论文
4.201067 明显不同。L connected 存档则和论文柔度精确一致。

3D 论文尺度静态复算使用纯 CPU Jacobi-CG，1,666 次达到相对残差
(9.99\times10^{-6})，墙钟 191.92 秒；观测峰值工作集约 1.90 GB，私有内存
约 2.98 GB。其柔度与 MATLAB CPU/ichol-PCG 静态 golden 的相对差为
(8.94\times10^{-10})。两端预条件器不同，所以 CG 迭代数不应直接比较；
状态量吻合才是这里的验证目标。

## 从论文初值重新优化

同一网格、几何、材料、投影参数、变量缩放、0.1 移动限和停止准则下：

| 案例 | 本实现 | 论文 | 柔度相对差 |
|---|---:|---:|---:|
| MBB | 88 次，C=4.256798501，vf=0.449988266 | 88 次，C=4.201067 | +1.3266% |
| L 型 | 83 次，C=2.846612664，vf=0.299969814 | 64 次，C=2.846072 | +0.0190% |

这说明方法和终态尺度得到复现，但不支持“逐迭代完全复现”的更强声明。原因是
论文发布时仓库没有锁定 MMA 源码；当前 MATLAB 仓库又有两套被路径顺序选择的
`mmasub.m`：默认 root 版和 GPL 子目录版会给出不同 MBB 结果。MATLAB 版本、
稀疏求解末位差异和非凸问题的渐近线趋势判定还会继续放大微差。

作为交叉检查，MATLAB R2023b 当前仓库的 MBB 结果是：

- 默认 root MMA：135 次，C=4.296627982；
- 强制 GPL MMA：97 次，C=4.259773894；
- 论文 R2019b：88 次，C=4.201067。

本实现结果与 GPL 路径的柔度更接近，但使用的是独立单约束 MMA 求解器，不能
据此断言论文采用了哪一个具体文件。

## PyGPTO 审计结论

PyGPTO 的投影、FE 和解析灵敏度核心可信，但固定提交不能在现代 SciPy 中原样
跑通论文三例：

- BC 中浮点 DOF 索引会被现代稀疏矩阵拒绝；
- `cg(tol=...)` API 已改为 `rtol`/`atol`，且 `maxiter` 被设置成 float；
- MMA 迭代编号偏一，第二次更新即提前切换渐近线；
- MBB 半径界/移动限和 L 型移动限偏离论文；
- L 网格坐标被截成四位小数；
- finite-difference、restart、trust-constr callback 和 2D VTK 路径存在确定性故障；
- `use_gpu` 没有等价 GPU 求解实现。

因此本项目没有直接修补并继续扩展 PyGPTO 的全局字典架构，而是保留其可信的
数学口径，用显式对象、整数索引、现代 CG API、测试和固定数据重新实现。

## 验证命令

```powershell
python -m pytest -q
gpto analyze mbb2d --profile paper --design initial
gpto analyze mbb2d --profile paper --design reference
gpto analyze lbracket2d --profile paper --design initial
gpto analyze lbracket2d --profile paper --design reference
gpto gradient-check lbracket2d --profile smoke
gpto optimize mbb2d --profile paper
gpto optimize lbracket2d --profile paper
gpto analyze cantilever3d --profile paper --design reference
```

数值结果保存在 `results/`。该目录默认被 Git 忽略，避免把大体积 VTK/NPZ 当作
源码提交；已验证数值另固化在 `validation/verified_results.json`，测试中的
golden 和包内小型参考几何也随项目分发。
