# 方法、代码对应与论文勘误

本文档采用论文的符号，但在公式排版与作者 MATLAB 实现不一致时，以后者和
有限差分验证为准。实现入口依次是：

```text
CaseDefinition
  -> DesignMap
  -> project_geometry
  -> FiniteElementModel.analyze
  -> ProjectionResult.local_vjp
  -> SingleConstraintMMA
```

对应源码为 `cases.py`、`problem.py`、`geometry.py`、`finite_elements.py` 和
`optimizer.py`。

## 1. 显式几何与投影

第 (b) 根构件由端点 \(\mathbf x_{1b},\mathbf x_{2b}\)、半径 (r_b) 与尺寸
变量 \(\alpha_b\) 描述。对单元质心 \(\mathbf x_e\)，首先求它到有限线段的
最短距离 (d_{be})：投影落在首端之外、线段内部和末端之外时分别使用
首端点距离、轴线正交距离和末端点距离。

内部为正的有符号距离是

\[
\phi_{be}=r_b-d_{be},\qquad \xi_{be}=\phi_{be}/r_e,
\]

其中采样窗口半径按单元体积自动设置：

\[
r_e=\frac{\sqrt d}{2}v_e^{1/d}.
\]

二维圆冠与三维球冠投影为

\[
\widetilde H_{2D}(\xi)=
\frac{\pi-\arccos\xi+\xi\sqrt{1-\xi^2}}{\pi},
\]

\[
\widetilde H_{3D}(\xi)=\frac12+\frac34\xi-\frac14\xi^3,
\]

只在 \(|\xi|<1\) 使用该多项式/圆冠表达式；两侧分别严格取 0 和 1。
尺寸作用后的构件密度为 (x_{be}=\alpha_b\rho_{be})。

## 2. 惩罚与构件合并

体积链使用未惩罚 (x_{be})，刚度链默认使用 SIMP：

\[
\widehat\rho_{be}=x_{be}^{q},\qquad q=3.
\]

论文默认 modified p-norm：

\[
\rho_e=
\left[\rho_{min}^{p}+(1-\rho_{min}^{p})
\sum_b\widehat\rho_{be}^{p}\right]^{1/p},
\quad \rho_{min}=0.01,\ p=8.
\]

它不是严格有界并集；多个构件重叠时 (ho_e>1) 是论文代码的预期行为，
因此 paper 模式不裁剪密度。项目另提供 modified p-mean、上下 KS 与概率并集，
但这些都属于实验性偏离，不能用于声称复现论文数值。

## 3. 有限元与优化问题

二维使用单位厚度 Q4 平面应力，三维使用 H8 各向同性实体，均采用每个方向
两点 Gauss 积分。单元密度在质心处取常值：

\[
\mathbf K_e=\rho_e^K\mathbf K_e^0,
\qquad \mathbf K(\mathbf z)\mathbf u=\mathbf f.
\]

目标和约束分别是

\[
c=\mathbf f^T\mathbf u,
\qquad
g=\frac{\sum_e v_e\rho_e^V}{\sum_e v_e}-v_f^*\le0.
\]

刚度链 (ho^K) 使用 SIMP3，体积链 (ho^V) 不惩罚。二者不能共用同一
密度数组。

## 4. 解析灵敏度

柔度对刚度密度的导数是

\[
\frac{\partial c}{\partial\rho_e^K}
=-\mathbf u_e^T\mathbf K_e^0\mathbf u_e,
\]

体积分数的单元权重是 (v_e/\sum v_e)。`ProjectionResult.local_vjp`
将这些权重反传经过聚合、惩罚、Heaviside 和点到线段距离；`DesignMap`
再完成端点共享累加和物理变量到 ([0,1]) 变量的链式缩放。

设计变量数为

\[
n_d n_p + 2n_b,
\]

顺序是全部点坐标、全部尺寸变量、全部半径。MBB、L 型和 3D 案例分别为
192、66 和 128 个变量。

## 5. MMA 口径

三例都只有一个体积约束。本项目直接求解 MMA 的单约束可分离倒数近似：给定
对偶乘子后，逐变量闭式求原变量，再用二分求对偶乘子。它避免复制 PyGPTO
所带 GPL `MMA.py`，但和 Svanberg primal-dual barrier 在末位数上并非逐步相同。

默认设置与论文/当前 MATLAB 输入对齐：缩放变量移动限 0.1、KKT 容差
(10^{-4})，MBB/L/3D 最大迭代数分别为 300/300/200。

## 6. 论文中需要更正或限定的地方

| 位置 | 论文文字/公式 | 本实现采用 |
|---|---|---|
| Eq. (3) | 二维圆冠根号项符号排错 | 上述正确圆冠式 |
| Eq. (8) | 写成 (d-r) | (phi=r-d)，内部为正 |
| Eq. (14)/(30) | 末端分支条件排错 | 轴向坐标 (≥) 线段长度 |
| Eq. (29) | 端点与半径导数符号反向 | (-\partial d/\partial x)，(\partial\phi/\partial r=1) |
| Eq. (22) | 载荷项多出表面梯度 | 标准边界载荷/离散载荷向量 |
| 设计变量数 | (2(n_dn_p+n_b)) | (n_dn_p+2n_b) |
| Eq. (6) 说明 | 暗示聚合结果不超过 1 | 重叠时可超过 1，不裁剪 |
| 域约束 | 暗示端点始终留在非凸域 | 上游只用包围盒坐标界；L 缺口不被额外约束 |

上述选择均由固定版本 MATLAB 源码和解析/有限差分测试交叉验证。

## 7. 上游代码到本实现的对应关系

| 功能 | MATLAB GPTO | PyGPTO | 本实现 |
|---|---|---|---|
| 输入与初始几何 | `input_files/*/inputs_*.m`, `initial_*.m` | `input_files/*/*.py` | `cases.py`, `data/*.npz` |
| 网格 | `generate_mesh.m`, Gmsh 导出 `.m` | `FE_routines.py`, Gmsh 导出 `.py` | `mesh.py` |
| 点到杆距离 | `compute_bar_elem_distance.m` | `geometry_projection.py` | `segment_distance_and_gradient` |
| 投影/惩罚/聚合 | `project_element_densities.m`, `smooth_max.m` | `geometry_projection.py` | `project_geometry`, `aggregate`, `penalize` |
| Q4/H8 与线性求解 | `FE_routines/*.m` | `FE_routines.py` | `finite_elements.py` |
| 柔度/体积及灵敏度 | `compute_compliance.m`, `compute_volume_fraction.m` | `functions.py` | `problem.py` |
| MMA | `runmma.m`, 两套 `mmasub.m` | `optimization.py`, GPL `MMA.py` | 独立 `optimizer.py` |
| 有限差分 | `utilities/fd_*.m` | `utilities.py`（当前损坏） | `GPTOProblem.gradient_check` |
| VTK/绘图 | `writevtk.m`, `plotting/*.m` | `plotting.py` | `artifacts.py`, `plotting.py` |

这里的 MATLAB/Python 文件名均指报告顶部锁定的 commit，而不是随时间变化的
默认分支。
