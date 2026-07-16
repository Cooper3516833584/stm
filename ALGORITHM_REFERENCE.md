# Cooper_drone 自主飞行算法参考手册

> **文献检索导向**：本文档详细描述 Cooper_drone 项目中实现的全部核心算法，包括数学公式、伪代码流程及推荐参考文献，便于在学术数据库中检索相关研究。
>
> **目标平台**：MYiR MYD-LD25X (STM32MP257 Cortex-A35 双核 2GB RAM, OpenSTLinux v6.0 aarch64)
> **传感器配置**：LDROBOT D500 单线激光雷达 ×2 + USB 摄像头 ×2 + 匿名凌霄飞控

---

## 目录

1. [候选方向管状碰撞检测避障](#1-候选方向管状碰撞检测避障)
2. [YOLO11-seg 道路分割与中线提取](#2-yolo11-seg-道路分割与中线提取)
3. [步进状态机任务调度](#3-步进状态机任务调度)
4. [多重安全仲裁器](#4-多重安全仲裁器)
5. [霍夫直线变换雷达 SLAM 位姿估计](#5-霍夫直线变换雷达-slam-位姿估计)
6. [ICPM-SVD 迭代最近点匹配](#6-icpm-svd-迭代最近点匹配)
7. [势场法路径规划](#7-势场法路径规划)
8. [五次多项式轨迹生成](#8-五次多项式轨迹生成)
9. [PID 闭环导航与传感器融合](#9-pid-闭环导航与传感器融合)
10. [单主路后处理](#10-单主路后处理)
11. [摄像头几何标定与偏移补偿](#11-摄像头几何标定与偏移补偿)
12. [D500 雷达数据包解析](#12-d500-雷达数据包解析)
13. [异步数据记录系统](#13-异步数据记录系统)

---

## 1. 候选方向管状碰撞检测避障

**实现位置**：[RelativeGoalNavigator.py](FlightController/Solutions/RelativeGoalNavigator.py)

### 1.1 问题建模

在无人机机体坐标系下，给定融合雷达点云 $\mathcal{P} = \{\mathbf{p}_i = (x_i, y_i)\}$，寻找前向扇区内的最优无碰航向角 $\theta^*$，输出速度指令 $(v_x, \dot{\psi})$，且满足 $v_y \equiv 0$（仅沿机体 x 轴前进，不输出侧向速度）。

### 1.2 数学模型

#### 候选方向生成

在扫描视场角 $\Phi = 150°$（可配）内以步长 $\Delta\theta = 2°$ 生成 $N$ 个候选方向：

$$\Theta = \left\{-\frac{\Phi}{2}, -\frac{\Phi}{2} + \Delta\theta, \ldots, 0, \ldots, \frac{\Phi}{2} - \Delta\theta, \frac{\Phi}{2}\right\}$$

确保 $0°$ 方向始终被包含。默认参数下 $N = 75$ 个候选方向。

#### 前向扫描点过滤

对原始点云 $\mathcal{P}$ 做三条件过滤：

$$\mathcal{P}_{\text{scan}} = \left\{\mathbf{p}_i \in \mathcal{P} \;\middle|\;
\begin{aligned}
&\|\mathbf{p}_i\| > d_{\min} = 10\text{cm} \quad\text{(滤除雷达壳体反射)} \\
&\|\mathbf{p}_i\| \leq \sqrt{L^2 + C^2} \quad\text{(前向感兴趣区域)} \\
&|\arctan2(y_i, x_i)| \leq \Phi/2 \quad\text{(仅前向扇区)}
\end{aligned}
\right\}$$

其中 $L = 220\text{cm}$ 为前瞻距离，$C = 80\text{cm}$ 为避障间隙。

#### 管状区域碰撞检测 (Tube-based Collision Check)

对于候选方向 $\theta$，定义单位方向向量 $\mathbf{u} = (\cos\theta, \sin\theta)$ 和法向量 $\mathbf{n} = (-\sin\theta, \cos\theta)$。将点云投影到轴向和法向：

$$\begin{aligned}
a_i &= \mathbf{p}_i \cdot \mathbf{u} \quad\text{(轴向投影)} \\
l_i &= |\mathbf{p}_i \cdot \mathbf{n}| \quad\text{(法向偏移)}
\end{aligned}$$

管状区域定义为：

$$\mathcal{T}(\theta) = \left\{\mathbf{p}_i \;\middle|\; d_{\min} < a_i \leq L \;\land\; l_i \leq C\right\}$$

该方向的管状间隙为：

$$d_{\text{tube}}(\theta) = \begin{cases}
\min\{a_i \mid \mathbf{p}_i \in \mathcal{T}(\theta)\}, & \mathcal{T}(\theta) \neq \emptyset \\
L, & \mathcal{T}(\theta) = \emptyset
\end{cases}$$

方向可通过性判定：

$$\text{allowed}(\theta) = d_{\text{tube}}(\theta) > C$$

#### 综合代价函数

对于每个可通过方向，计算三项代价的加权和：

$$J(\theta) = \underbrace{|\text{wrap}(\theta - \theta_{\text{goal}})|}_{\text{heading cost}} + \underbrace{J_{\text{clear}}(d_{\text{tube}})}_{\text{clearance cost}} + \underbrace{|\text{wrap}(\theta - \theta_{\text{last}})| \cdot w_s}_{\text{switch cost}}$$

其中间隙代价为线性斜坡函数：

$$J_{\text{clear}}(d) = \begin{cases}
0, & d \geq D_{\text{avoid}} \\
\frac{D_{\text{avoid}} - d}{D_{\text{avoid}} - C} \cdot w_c, & C < d < D_{\text{avoid}}
\end{cases}$$

$D_{\text{avoid}} = 150\text{cm}$ 为避障起始距离，$w_c = 120$ 为间隙代价权重，$w_s = 0.15$ 为切换代价权重。

#### 方向选择与迟滞

**最优选择**：
$$\theta^* = \arg\min_{\theta: \text{allowed}(\theta)} J(\theta)$$

**切换迟滞**：若上次选择方向 $\theta_{\text{last}}$ 的代价值 $J(\theta_{\text{last}}) \leq J(\theta^*) + 5.0$（切换容忍度），则保持 $\theta_{\text{last}}$。此机制避免连续帧间方向振荡。

**无可通过方向时**：选择 $d_{\text{tube}}$ 最大的方向，按偏航搜索角度进行原地旋转。

#### 阻塞释放迟滞

当系统进入阻塞状态（$\text{blocked} = \text{true}$），需要 $0°$ 方向间隙恢复至 $C_{\text{release}} = 90\text{cm} > C = 80\text{cm}$ 才解除阻塞。这种 hysteresis 防止在临界距离附近反复振荡。

#### 偏航控制律

采用比例控制器，带最小速率限制：

$$\dot{\psi} = \begin{cases}
0, & |\theta^*| \leq \theta_{\text{stop}} = 3° \\
\text{clip}\left(K_p \cdot \theta^*, \pm\dot{\psi}_{\max}\right), & \text{otherwise}
\end{cases}$$

$$|\dot{\psi}| < \dot{\psi}_{\min} \Rightarrow \dot{\psi} = \text{sgn}(\dot{\psi}) \cdot \dot{\psi}_{\min}$$

其中 $K_p = 0.5$，$\dot{\psi}_{\max} = 25°/s$，$\dot{\psi}_{\min} = 6°/s$。

#### 速度成形

前进速度根据前向间隙线性内插：

$$v_x(d_{\text{front}}) = \begin{cases}
0, & d_{\text{front}} \leq C \\
v_{\min} + \frac{d_{\text{front}} - C}{D_{\text{avoid}} - C} \cdot (v_{\text{cruise}} - v_{\min}), & C < d_{\text{front}} < D_{\text{avoid}} \\
v_{\text{cruise}}, & d_{\text{front}} \geq D_{\text{avoid}}
\end{cases}$$

默认 $v_{\min} = 8\text{cm/s}$，$v_{\text{cruise}} = 20\text{cm/s}$。

### 1.3 伪代码

```
function update(obstacles_body_cm):
    points ← filter_front_scan(obstacles_body_cm)

    for each angle in candidate_angles:
        tube_clearance ← tube_collision_check(points, angle)
        cost ← heading_cost + clearance_cost + switch_cost
        evaluations.append(angle, tube_clearance > obstacle_clearance, cost)

    allowed ← filter(evaluations, allowed=true)

    if allowed is empty:
        return blocked_no_path or yaw_search

    best ← argmin(allowed, cost)
    best ← apply_switch_hysteresis(best, last_selected_angle)

    if |best.angle| > align_start:
        return turn_in_place(best.angle)

    front ← tube_check(0°)
    if not front.allowed or not can_move_forward:
        return hold_or_turn

    return forward(speed_from_clearance(front.clearance))
```

### 1.4 关键参数汇总

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `scan_fov_deg` | 150° | 扫描视场角 |
| `candidate_step_deg` | 2° | 候选方向步长 |
| `obstacle_clearance_cm` | 80 | 硬避障间隙 |
| `clearance_release_cm` | 90 | 阻塞释放阈值 |
| `lookahead_cm` | 220 | 前瞻距离 |
| `avoid_begin_distance_cm` | 150 | 避障起始/速度成形起始距离 |
| `align_start_deg` | 10 | 转向触发角 |
| `align_stop_deg` | 3 | 转向停止角 |
| `yaw_kp` | 0.5 | 偏航 P 增益 |

### 1.5 推荐参考文献

- **Tube-based collision checking**: Borenstein, J. & Koren, Y. (1991). The Vector Field Histogram — Fast Obstacle Avoidance for Mobile Robots. *IEEE Transactions on Robotics*, 7(3), 278–288.
- **Direction candidate search**: Fox, D., Burgard, W. & Thrun, S. (1997). The Dynamic Window Approach to Collision Avoidance. *IEEE Robotics & Automation Magazine*, 4(1), 23–33.
- **Hysteresis in obstacle avoidance**: Minguez, J. & Montano, L. (2004). Nearness Diagram (ND) Navigation: Collision Avoidance in Troublesome Scenarios. *IEEE Trans. Robotics*, 20(1), 45–59.

---

## 2. 道路语义分割与中线提取

**实现位置**：[road_perception.py](road_perception.py)

### 2.1 问题建模

给定 640×480 BGR 图像帧，默认使用 `new_road_seg_v3_final_fp32.nb`
在 VIP9000 NPU 上进行二类语义分割，从道路 mask 中提取中线点序列，输出
像素偏差 $e_{\text{px}}$、中线角度 $\alpha$ 及道路状态。系统只输出当前主路和
`single` 状态；`fast-main` 使用稀疏低分辨率 mask，`full` 保留全分辨率主路几何。
原 YOLO11n-seg 128×128 ONNX 实例分割实现保留为 CPU 回退，但同样不做分叉处理。

### 2.2 数学模型

#### 默认 NPU 语义分割管线

1. **直接缩放**：整帧缩放至 $256 \times 256$，与训练变换保持一致
2. **归一化**：`BGR → RGB`，`[0,255] → [0,1]` float32，NCHW
3. **推理**：STAI MPU 加载 `.nb`，使用 VIP9000 NPU
4. **快速后处理**：解析 `logits [1,2,256,256]`，将 class 1 mask 缩放到
   192×144，使用 3×3 单次闭运算和连通域保留底部主路
5. **稀疏中线**：仅扫描 mask 下半部分，垂直步长为 2；中线点、宽度和误差按
   原图/工作 mask 比例恢复到 640×480 像素坐标
6. **单主路输出**：误差、角度和宽度只拟合一次，直接输出 `centerline_points`；
   `branches` 保持为空，不执行分叉检测、候选分支构造和第二次 mask 清理

实现时必须将 `transpose` 后的 NCHW 输入转为 C-contiguous 数组；`stai_mpu`
按连续内存读取应用缓冲，不处理 NumPy strides。道路 mask 的逐行区间采用 NumPy
向量化提取。`full` 模式仅提高 mask 几何分辨率，不恢复路口判断或分支构造；
CPU YOLO 路径也使用相同的单主路输出契约。

#### CPU 回退 YOLO 管线

1. **Letterbox 预处理**：保持宽高比的缩放 + 灰色填充至 $320 \times 320$
2. **归一化**：`BGR → RGB`，`[0,255] → [0,1]` 浮点
3. **推理**：ONNX Runtime，固定 XNNPACK/CPU，避免旧 YOLO 图触发 VSINPU 已知崩溃
4. **后处理**：解析输出张量 `output0 [1, C, N]`（检测头）+ `output1 [1, M, H, W]`（mask prototype）

#### Mask 解码

对每个通过 NMS 的检测框 $k$：

$$\text{mask}_k = \sigma\left(\mathbf{c}_k^\top \cdot \mathbf{P}_{\text{flat}}\right)$$

其中 $\mathbf{c}_k \in \mathbb{R}^M$ 为第 $k$ 个检测的 mask coefficients，$\mathbf{P}_{\text{flat}} \in \mathbb{R}^{M \times (H_p W_p)}$ 为展平的 prototype masks，$\sigma$ 为 sigmoid 函数。

Mask 经裁剪到 bbox 区域 → 裁掉 letterbox 填充 → resize 回原始分辨率 → 二值化（阈值 0.5）。

#### Mask 清理

```
形态学闭运算 (5×5 kernel, 2 iterations) → 连通域分析 → 保留最大面积且触碰画面底部的连通域
```

#### 中线点提取

从画面底部向上逐行扫描 mask，每行提取连通区间，以一阶指数平滑（$\alpha = 0.65$）跟踪中线横坐标：

$$c_x^{(y)} = \alpha \cdot \frac{l + r}{2} + (1 - \alpha) \cdot c_x^{(y+1)}$$

其中 $l, r$ 为距离上一行中线最近的区间左右端点。

#### 中线角度的线性拟合

在中部区域（画面 72%–98% 高度）的中线点上做最小二乘直线拟合：

$$\begin{bmatrix} y_1 & 1 \\ \vdots & \vdots \\ y_n & 1 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}$$

中线角度 $\alpha = \arctan2(1, -a)$，$a$ 为斜率（像素坐标系 x = a·y + b）。

#### 像素偏差

取画面底部（82%–96% 高度）中线点横坐标的中位数 $c_x^{\text{bottom}}$：

$$e_{\text{px}} = c_x^{\text{bottom}} - \frac{W}{2}$$

### 2.3 单主路约束

实验道路无分叉，中心线扫描逐行选择与上一行中心最近的有效区间。该约束避免保存
全量行区间、路口宽度增长判断、S-curve 穿越平滑以及候选分支拟合。

### 2.4 推荐参考文献

- **YOLOv11 Segmentation**: Jocher, G. et al. (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
- **YOLACT / Proto-mask**: Bolya, D. et al. (2019). YOLACT: Real-Time Instance Segmentation. *ICCV 2019*.
- **Centerline for visual road following**: Chen, Z. & Birchfield, S. (2009). Qualitative Vision-Based Path Following. *IEEE Trans. Robotics*, 25(3), 749–754.
- **Letterbox**: Redmon, J. & Farhadi, A. (2018). YOLOv3. arXiv:1804.02767.

---

## 3. 步进状态机任务调度

**实现位置**：[goal_nav_main.py](goal_nav_main.py) 函数 `_next_step_command()`

### 3.1 状态转移

```
PLAN ──(规划成功)──→ EXECUTE ──(超时/打断)──→ HOLD ──(等待结束)──→ PLAN
  │                     │                        │
  └──(无路径)──→ mission_done           (安全中断)→ HOLD
```

### 3.2 步进时长

$$\Delta t_{\text{step}} = \text{clip}\left(\frac{D_{\text{step}}}{v_x}, t_{\min}, t_{\max}\right)$$

默认：$D_{\text{step}} = 40\text{cm}$，$t_{\min} = 0.35\text{s}$，$t_{\max} = 2.0\text{s}$。

转向固定 0.45s，HOLD 固定 0.35s。

### 3.3 执行阶段打断条件

在 EXECUTE 阶段每循环重新评估当前指令是否仍然安全：

$$\text{should\_interrupt} = (v_x^{\text{active}} > 0) \land (v_x^{\text{live}} \leq 0 \lor \dot{\psi}^{\text{live}} > 0)$$

即：前进中出现新障碍需停车或转向时，不等步进超时立即打断。

### 3.4 推荐参考文献

- **Step-by-step re-planning**: LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press. (Chapter 14: Incremental Planning)
- **Reactive interrupt**: Brooks, R. A. (1986). A Robust Layered Control System for a Mobile Robot. *IEEE J. Robotics & Automation*, 2(1), 14–23.

---

## 4. 多重安全仲裁器

**实现位置**：[Safety.py](FlightController/Solutions/Safety.py)

### 4.1 硬故障检查（Hard Stop）

$$
\text{HARD\_STOP} \iff \bigvee\begin{cases}
\text{FC 未连接} \\
\text{非定点模式 (mode} \neq 2) \\
\text{雷达帧超时} > 0.5\text{s} \\
|\text{roll}| > 25° \\
|\text{pitch}| > 25° \\
\text{电池电压} < V_{\min} \text{（可选）}
\end{cases}
$$

硬故障时输出全零速度指令。

### 4.2 障碍物分级限速

令 $d_{\text{front}}$ 为前方走廊最近障碍物距离：

$$\text{action} = \begin{cases}
\text{STOP} \; (v_x \leftarrow 0), & d_{\text{front}} \leq 80\text{cm} \\
\text{SLOW} \; (v_x \leftarrow \min(v_x, 12\text{cm/s})), & 80 < d_{\text{front}} \leq 150\text{cm} \\
\text{OK}, & d_{\text{front}} > 150\text{cm}
\end{cases}$$

### 4.3 侧向限位

$$\text{left\_blocked}: v_y > 0 \land d_{\text{left}} \leq 45\text{cm} \Rightarrow v_y \leftarrow 0$$
$$\text{right\_blocked}: v_y < 0 \land d_{\text{right}} \leq 45\text{cm} \Rightarrow v_y \leftarrow 0$$

### 4.4 速度钳位

所有输出分量经硬限制钳位：$v_x \leq 35\text{cm/s}$, $v_y \leq 25\text{cm/s}$, $v_z \leq 20\text{cm/s}$, $\dot{\psi} \leq 30°/s$。

### 4.5 推荐参考文献

- **Safety monitors in UAV**: Fravolini, M. L. et al. (2018). Structural Health Monitoring for UAV. In *Safety and Reliability*, 38(1).
- **Runtime verification**: Desai, A. et al. (2019). SOTER: Programming Safe Robotics. *ICRA 2019*.

---

## 5. 霍夫直线变换雷达 SLAM 位姿估计

**实现位置**：[Radar_SLAM.py](FlightController/Solutions/Radar_SLAM.py) 函数 `radar_resolve_rt_pose()`

### 5.1 问题建模

在结构化室内环境中（矩形房间），从 2D 激光雷达点云栅格化图像中检测墙壁直线，利用正交几何关系解算无人机在房间内的相对位姿 $(x, y, \psi)$。

### 5.2 数学模型

#### 栅格化与形态学预处理

点云投影到 $W \times H$ 灰度图像（默认 1000×1000），每个点的灰度值为距离毫米数。预处理：

$$\text{img}' = \text{erode}(\text{dilate}(\text{img}, K_d), K_e)$$

膨胀核 $9 \times 9$ 椭圆，腐蚀核 $5 \times 5$ 椭圆。

#### 霍夫直线检测

$$\text{lines} = \text{HoughLinesP}(\text{img}', 1, 1°, \tau=80, \ell_{\min}=60, g_{\max}=200)$$

#### 墙壁分类（3 种方法，当前使用 Method 3）

对每条直线计算中点 $(m_x, m_y)$ 和线段角度 $\alpha$：

- **右侧墙** (right wall): $(|\alpha| > 45° \lor |\alpha| < -45°) \land m_x > x_0$
- **后方墙** (back wall): $(|\alpha| < 45° \land |\alpha| > -45°) \land m_y > y_0$

#### 位姿解算

对右侧墙（应平行于飞行方向）：选择距离图像中心最近的直线，其到中心点的距离即为 Y 坐标：

$$y = d_{\text{right}}, \quad \psi_1 = -\alpha_{\text{right}}$$

对后方墙（应垂直于飞行方向）：

$$x = d_{\text{back}}, \quad \psi_2 = -\alpha_{\text{back}} + 90°$$

角度融合（正交约束）：若 $|\psi_1 - \psi_2| \leq 30°$，则 $\psi = (\psi_1 + \psi_2) / 2$；否则角度不可靠、舍弃。

#### 低通滤波

对 $(x, y, \psi)$ 用一阶低通（$\alpha = 0.1$）平滑输出，用于后续 ICPM 配准。

### 5.3 推荐参考文献

- **Hough Transform**: Duda, R. O. & Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. *Comm. ACM*, 15(1), 11–15.
- **2D laser SLAM with line features**: Nguyen, V. et al. (2007). A Comparison of Line Extraction Algorithms using 2D Range Data. *Robotics and Autonomous Systems*, 55(5), 393–403.
- **Orthogonal assumption**: Ahn, S. J. et al. (2001). Least-Squares Orthogonal Distances Fitting. *Pattern Recognition*, 34(12), 2447–2457.

---

## 6. ICPM-SVD 迭代最近点匹配

**实现位置**：[Radar_SLAM.py](FlightController/Solutions/Radar_SLAM.py) 类 `ICPM`

### 6.1 问题建模

给定模板点云 $\mathcal{P} = \{\mathbf{p}_j\}$（前一帧）和当前点云 $\mathcal{Q} = \{\mathbf{q}_i\}$（当前帧），求刚体变换 $(\mathbf{R}, \mathbf{t})$ 使得点云配准：

$$(\mathbf{R}^*, \mathbf{t}^*) = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_i \|\mathbf{R}\mathbf{q}_{\text{match}(i)} + \mathbf{t} - \mathbf{p}_i\|^2$$

### 6.2 数学模型

#### Step 1: 最近邻关联

$$\text{match}(i) = \arg\min_j \|\mathbf{q}_i - \mathbf{p}_j\|$$

累积误差：$E = \sum_i \|\mathbf{q}_{\text{match}(i)} - \mathbf{p}_i\|$

#### Step 2: SVD 运动估计

去中心化：

$$\bar{\mathbf{p}} = \frac{1}{N}\sum \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum \mathbf{q}_{\text{match}(i)}$$
$$\tilde{\mathbf{p}}_i = \mathbf{p}_i - \bar{\mathbf{p}}, \quad \tilde{\mathbf{q}}_i = \mathbf{q}_{\text{match}(i)} - \bar{\mathbf{q}}$$

交叉协方差矩阵：

$$\mathbf{W} = \sum_i \tilde{\mathbf{q}}_i \tilde{\mathbf{p}}_i^\top$$

SVD 分解：

$$\mathbf{W} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

最优旋转：

$$\mathbf{R}^* = \mathbf{V} \mathbf{U}^\top$$

（注：代码中实际为 $\mathbf{R} = (\mathbf{U}\mathbf{V}^\top)^\top = \mathbf{V}\mathbf{U}^\top$，与学术标准一致）

最优平移：

$$\mathbf{t}^* = \bar{\mathbf{p}} - \mathbf{R}^* \bar{\mathbf{q}}$$

#### Step 3: 迭代与收敛

重复 Step 1–2，更新 $\mathcal{Q} \leftarrow \mathbf{R}\mathcal{Q} + \mathbf{t}$，直到误差变化 $\Delta E < 10^{-4}$ 或达到最大迭代次数 100。

欧拉角提取：$\psi = \arcsin(R_{01})$

### 6.3 推荐参考文献

- **ICP**: Besl, P. J. & McKay, N. D. (1992). A Method for Registration of 3-D Shapes. *IEEE Trans. PAMI*, 14(2), 239–256.
- **SVD for rotation**: Arun, K. S., Huang, T. S. & Blostein, S. D. (1987). Least-Squares Fitting of Two 3-D Point Sets. *IEEE Trans. PAMI*, 9(5), 698–700.
- **Umeyama's refinement**: Umeyama, S. (1991). Least-Squares Estimation of Transformation Parameters Between Two Point Patterns. *IEEE Trans. PAMI*, 13(4), 376–380.

---

## 7. 势场法路径规划

**实现位置**：[PathPlanner.py](FlightController/Solutions/PathPlanner.py) 类 `PFBPP`

### 7.1 问题建模

在 2D 占据栅格地图中，利用**人工势场法**（Artificial Potential Field）从起点 $\mathbf{s}$ 到目标点 $\mathbf{g}$ 规划无碰路径，输出离散路点序列。

### 7.2 数学模型

#### 势场构造

在 $(x_w \times y_w)$ 网格（默认分辨率 $0.5\text{m}$）上计算每个格点 $(x, y)$ 的势能：

$$U(x, y) = U_{\text{attr}}(x, y) + U_{\text{rep}}(x, y)$$

**吸引势**（二次势）：

$$U_{\text{attr}}(x, y) = \frac{1}{2} k_a \cdot \sqrt{(x - g_x)^2 + (y - g_y)^2}$$

$k_a = 5.0$ 为吸引增益。

**斥力势**（FIRAS 形式）：

$$U_{\text{rep}}(x, y) = \begin{cases}
\frac{1}{2} k_r \left(\frac{1}{d_{\text{obs}}} - \frac{1}{d_0}\right)^2, & d_{\text{obs}} \leq d_0 \\
0, & d_{\text{obs}} > d_0
\end{cases}$$

$k_r = 100.0$ 为斥力增益，$d_0 = 1.0\text{m}$ 为机器人半径（影响范围）。

#### 梯度下降路径搜索

从起始格点 $(i_x, i_y)$ 出发，每次取 8 邻域中势能最小的格点为下一步：

$$(i_x, i_y) \leftarrow \arg\min_{(i,j) \in \mathcal{N}_8} U(i_x + \Delta i, i_y + \Delta j)$$

直到距离目标 $\sqrt{(x - g_x)^2 + (y - g_y)^2} < \text{grid\_size}$。

#### 局部极小值检测与逃逸

维护最近 $N = 4$ 步的格点历史。若出现重复格点（振荡），则从当前位置回退 $10$ 步，增大 $d_0$（每次 $+0.5\text{m}$）重试，最多重试 $10$ 次。

### 7.3 推荐参考文献

- **Potential Field**: Khatib, O. (1986). Real-Time Obstacle Avoidance for Manipulators and Mobile Robots. *IJRR*, 5(1), 90–98.
- **Local minima escape**: Barraquand, J. & Latombe, J.-C. (1991). Robot Motion Planning: A Distributed Representation Approach. *IJRR*, 10(6), 628–649.
- **Gradient descent path**: Ge, S. S. & Cui, Y. J. (2000). New Potential Functions for Mobile Robot Path Planning. *IEEE Trans. Robotics*, 16(5), 615–620.

---

## 8. 五次多项式轨迹生成

**实现位置**：[PathPlanner.py](FlightController/Solutions/PathPlanner.py) 类 `TrajectoryGenerator`

### 8.1 问题建模

给定起始位姿/速度/加速度和目标位姿/速度/加速度共 6 个约束，生成 $C^2$ 连续的光滑轨迹。

### 8.2 数学模型

三个轴 (x, y, z) 分别求解。以 x 轴为例：

$$x(t) = c_0 t^5 + c_1 t^4 + c_2 t^3 + c_3 t^2 + c_4 t + c_5$$

边界条件：

$$\begin{aligned}
x(0) = x_s, &\quad x(T) = x_g \\
\dot{x}(0) = v_s, &\quad \dot{x}(T) = v_g \\
\ddot{x}(0) = a_s, &\quad \ddot{x}(T) = a_g
\end{aligned}$$

线性系统求解 $\mathbf{A}\mathbf{c} = \mathbf{b}$：

$$\mathbf{A} = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 1 \\
T^5 & T^4 & T^3 & T^2 & T & 1 \\
0 & 0 & 0 & 0 & 1 & 0 \\
5T^4 & 4T^3 & 3T^2 & 2T & 1 & 0 \\
0 & 0 & 0 & 2 & 0 & 0 \\
20T^3 & 12T^2 & 6T & 2 & 0 & 0
\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix} x_s \\ x_g \\ v_s \\ v_g \\ a_s \\ a_g \end{bmatrix}$$

速度与加速度：

$$\begin{aligned}
\dot{x}(t) &= 5c_0 t^4 + 4c_1 t^3 + 3c_2 t^2 + 2c_3 t + c_4 \\
\ddot{x}(t) &= 20c_0 t^3 + 12c_1 t^2 + 6c_2 t + 2c_3
\end{aligned}$$

总时间 $T$ 由距离和期望巡航速度决定：$T = \|\mathbf{g} - \mathbf{s}\| / v_{\text{navi}}$。

### 8.3 应用场景

- **直线导航** `navigation_to_waypoint()`：直线路径的五次多项式轨迹
- **圆形巡航** `navigation_around_waypoint()`：离散圆周采样点以直线段连接，每段生成五次多项式

### 8.4 推荐参考文献

- **Quintic polynomial**: Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control*. 3rd ed., Pearson. (Chapter 7: Trajectory Generation)
- **Minimum-jerk trajectories**: Flash, T. & Hogan, N. (1985). The Coordination of Arm Movements. *J. Neuroscience*, 5(7), 1688–1703.

---

## 9. PID 闭环导航与传感器融合

**实现位置**：[Navigation.py](FlightController/Solutions/Navigation.py) 类 `Navigation`

### 9.1 问题建模

在 HOLD_POS（定点）飞控模式下，使用外部定位源（雷达 SLAM / Realsense T265 / 两者融合）做位置闭环，以 PID 控制器输出速度指令驱动飞机。

### 9.2 数学模型

#### PID 控制器

每个轴独立 PID：

$$u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de}{dt}$$

双轴平面导航使用相同 PID 参数，垂直高度和偏航各一组 PID。

#### 多组 PID 参数切换

| 模式 | $K_p$ | $K_i$ | $K_d$ | 适用场景 |
|------|-------|-------|-------|---------|
| default | 0.35 | 0.0 | 0.08 | 一般悬停 |
| navi | 1.4 | 0.0 | 0.02 | 轨迹跟踪 |
| hover | 0.65 | 0.0 | 0.02 | 精确悬停 |
| land | 0.85 | 0.0 | 0.02 | 降落对准 |

#### 多源定位模式

| 模式 | 定位源 | 融合策略 |
|------|--------|----------|
| `radar` | 雷达 SLAM 位姿 | 直接使用 |
| `rs` | Realsense T265 VIO | 直接使用 |
| `fusion` | T265 + 雷达 SLAM | 雷达每 $N$ 帧校准一次 T265 坐标原点 |
| `fusion-ros` | T265 + ROS 建图 | 外部建图模块提供坐标变换 |

#### 雷达→T265 坐标校准

从雷达 SLAM 位姿 $(x_r, y_r, \psi_r)$ 计算 T265 坐标系的平移偏移：

$$\begin{aligned}
\Delta z_{\text{t265}} &= -(x_r - x_{\text{base}}) / 100 \quad\text{→ 反向符号, 米} \\
\Delta x_{\text{t265}} &= -(y_r - y_{\text{base}}) / 100 \\
\Delta\psi_{\text{t265}} &= -\psi_r
\end{aligned}$$

（T265 坐标系：z 前向，x 右向，y 向上；匿名坐标系：x 前向，y 左向）

#### 轨迹跟踪

利用 PathPlanner 生成离散轨迹点序列，逐个设置为 PID 目标点，抵达阈值 $\epsilon_{\text{pos}} = 10\text{cm}$ 后切换到下一点：

$$\|\mathbf{p}_{\text{current}} - \mathbf{p}_{\text{target}}\|^2 \leq \epsilon_{\text{pos}}^2$$

### 9.3 推荐参考文献

- **PID control**: Åström, K. J. & Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning*. ISA.
- **Multi-sensor fusion for UAV**: Weiss, S. et al. (2012). Monocular Vision for Long-term MAV Navigation. *IROS 2012*.
- **Sensor calibration between frames**: Lv, J. et al. (2019). Observability-Aware Intrinsic and Extrinsic Calibration. *ICRA 2019*.

---

## 10. 单主路后处理

**实现位置**：[road_perception.py](road_perception.py) 函数
`_extract_centerline_and_intervals()`、`_extract_fast_main_centerline()`。

实验道路没有分叉，因此感知结果固定为 `single`：

1. 清理道路 mask，并从候选实例中选择与画面底部中心最相关的主路；
2. 仅对选中的主路提取一次中心线；
3. 直接计算像素误差、中线角度和道路宽度；
4. `branches=[]`、`selected_branch=None`、`branch_decision="disabled"`；
5. `RoadFollower` 直接使用 `corrected_pixel_error` 和 `centerline_angle`。

该流程删除了逐帧路口检测、分支构建、去重、分类、历史分支保持和二次 mask 扫描。
兼容字段仍保留在结果结构中，避免旧日志或外部读取器因字段缺失而崩溃。

---

## 11. 摄像头几何标定与偏移补偿

**实现位置**：[road_perception.py](road_perception.py) 函数 `compute_meters_per_pixel()`, `apply_camera_offset_compensation()`

### 11.1 逐行 m/pixel 计算

基于针孔模型与地面平面假设：

$$\theta(r) = \alpha + \beta \cdot \left(1 - \frac{2r}{H-1}\right)$$
$$D_{\text{ground}} = \frac{h}{\tan(\theta)}$$
$$\text{m/px}(r) = \frac{2 \cdot D_{\text{ground}} \cdot \tan(\text{HFOV}/2)}{W}$$

其中 $\alpha$ 为光轴倾角，$\beta$ 为半垂直视场角，$h$ 为飞行高度，$r$ 为距离画面底部的行号。

### 11.2 摄像头前向偏移补偿

摄像头相对机体中心的纵向位置为带符号的 $d_{\text{off}}$（机体前方为正、后方为负），导致航向偏差时路面中线产生视差：

$$\Delta e = \text{sign} \cdot \frac{d_{\text{off}}}{\text{m/px}} \cdot \tan(\psi_{\text{error}})$$

补偿值钳位于 $\pm 120\text{px}$。当前道路摄像头垂直向下安装，机体坐标为 $(x,y)=(-0.0787, 0)\text{m}$，故 $d_{\text{off}}=-0.0787\text{m}$。修正后的像素误差 $e' = e - \Delta e$ 用于控制；仍须提供实测 `meters_per_pixel_x` 才会启用该补偿。

### 11.3 软件白平衡

对偏色摄像头做乘性 BGR 通道校正：

$$\begin{bmatrix} B' \\ G' \\ R' \end{bmatrix} = \begin{bmatrix} B \cdot g_B \\ G \cdot g_G \\ R \cdot g_R \end{bmatrix}$$

### 11.4 推荐参考文献

- **Inverse perspective mapping**: Bertozzi, M. & Broggi, A. (1998). GOLD: A Parallel Real-Time Stereo Vision System. *IEEE Intelligent Systems*, 13(1), 30–39.
- **Camera forward offset**: Corke, P. (2017). *Robotics, Vision and Control*. 2nd ed., Springer.

---

## 12. D500 雷达数据包解析

**实现位置**：[LDRadar_Resolver.py](FlightController/Components/LDRadar_Resolver.py)

### 12.1 数据包结构

D500 雷达以 230400 baud 输出固定格式数据包：

| 字节位 | 内容 | 说明 |
|--------|------|------|
| 0–1 | Header | 帧起始标识 0x54 0x2C |
| 2 | VerLen | 版本 + 包长信息 |
| 3 | Speed | 转速 LSB (deg/s) |
| 4 | Speed | 转速 MSB |
| 5 | StartAngle | 起始角 LSB (0.01°) |
| 6 | StartAngle | 起始角 MSB |
| 7–48 | Data | 12 组测距点 (距离 mm + 置信度) |
| 49 | EndAngle | 终止角 LSB |
| 50 | EndAngle | 终止角 MSB |
| 51 | Timestamp | 时间戳 LSB (ms) |
| 52 | Timestamp | 时间戳 MSB |
| 53 | CRC | CRC8 校验 |

每包 54 字节，~300 包/秒。

### 12.2 点云重建

每个测距点的角度线性插值：

$$\theta_k = \theta_{\text{start}} + \frac{k}{11} \cdot (\theta_{\text{end}} - \theta_{\text{start}})$$

极坐标 → 笛卡尔坐标：

$$x_k = d_k \cos(\theta_k), \quad y_k = -d_k \sin(\theta_k)$$

以 Map_Circle 环形缓冲区存储 360° 完整扫描帧。

### 12.3 推荐参考文献

- **LIDAR data protocol**: LDROBOT. D500 Development Manual.
- **Point cloud from 2D LIDAR**: Quigley, M. et al. (2009). ROS: an open-source Robot Operating System. *ICRA Workshop*.

---

## 13. 异步数据记录系统

**实现位置**：[record_data.py](record_data.py) / [SessionRecorder.py](FlightController/Solutions/SessionRecorder.py)

### 13.1 设计原理

采用生产者-消费者模式：

- **主线程（生产者）**：以 10Hz 采样雷达点云 + 相机帧，推入队列
- **写入线程（消费者）**：从队列取出数据，压缩后写入 SD 卡

### 13.2 存储格式

| 数据类型 | 格式 | 文件 |
|----------|------|------|
| 雷达原始 bin 数据 | raw bytes | `radar_bins/{seq:06d}.bin` |
| 雷达点云坐标 | `np.savez_compressed` | `radar_points/{seq:06d}.npz` |
| 雷达元数据 | JSONL | `radar.jsonl` |
| 相机帧 | JPEG (cv2.imencode) | `frames/{seq:06d}.jpg` |
| 控制指令 | JSONL | `commands.jsonl` |
| 会话信息 | JSON | `session.json` |

### 13.3 关键参数

- 队列容量：60 条（~6 秒缓冲 at 10Hz）
- JPEG 质量：85
- 默认帧保存间隔：每 10 个循环 1 帧
- 输出目录：`/media/sdcard/recordings/`

### 13.4 推荐参考文献

- **Producer-consumer with bounded queue**: Hoare, C. A. R. (1978). Communicating Sequential Processes. *Comm. ACM*, 21(8).
- **Flight data logging**: Watkins, S. et al. (2010). Ten Years of UAV Data Logging. *AIAA Infotech*.

---

## 附录 A：项目文件索引

### 入口脚本
| 文件 | 功能 |
|------|------|
| [goal_nav_main.py](goal_nav_main.py) | 雷达避障自主导航入口 |
| [road_follow_main.py](road_follow_main.py) | 视觉道路循线入口 |
| [record_data.py](record_data.py) | 纯数据记录器 |

### 核心算法
| 文件 | 算法 |
|------|------|
| [RelativeGoalNavigator.py](FlightController/Solutions/RelativeGoalNavigator.py) | 管状碰撞检测避障 (§1) |
| [RoadFollower.py](FlightController/Solutions/RoadFollower.py) | P 控制道路跟随 (§2) |
| [road_perception.py](road_perception.py) | YOLO11-seg 道路感知 (§2, §10, §11) |
| [Safety.py](FlightController/Solutions/Safety.py) | 多重安全仲裁 (§4) |
| [Radar_SLAM.py](FlightController/Solutions/Radar_SLAM.py) | 霍夫 SLAM + ICPM (§5, §6) |
| [PathPlanner.py](FlightController/Solutions/PathPlanner.py) | 势场法 + 五次多项式 (§7, §8) |
| [Navigation.py](FlightController/Solutions/Navigation.py) | PID 闭环导航 + 传感器融合 (§9) |

### 硬件驱动
| 文件 | 功能 |
|------|------|
| [LDRadar_Driver.py](FlightController/Components/LDRadar_Driver.py) | 乐动 D500 雷达驱动 |
| [LDRadar_Resolver.py](FlightController/Components/LDRadar_Resolver.py) | D500 数据包解析 (§12) |
| [MultiRadar.py](FlightController/Components/MultiRadar.py) | 双雷达融合管理 |
| [CameraSource.py](FlightController/Components/CameraSource.py) | V4L2 摄像头封装 |

### 飞控协议栈
| 文件 | 层 |
|------|-----|
| [Base.py](FlightController/Base.py) | 串口通信基类 + Byte_Var |
| [Protocal.py](FlightController/Protocal.py) | 协议命令层 |
| [Application.py](FlightController/Application.py) | 应用接口层 |
| [FCConnector.py](FlightController/Components/FCConnector.py) | 飞控连接工厂 |

---

## 附录 B：关键技术参数速查

| 参数 | 值 |
|------|-----|
| 雷达型号 / 转速 | LDROBOT D500 / 590–600 RPM (~10Hz) |
| 雷达数据包速率 | ~300 包/秒, 12 点/包 |
| 雷达扫描范围 | 360°, 距离 0.1–15m |
| 摄像头分辨率 | 640×480, 30 FPS |
| YOLO 推理输入 | 320×320 |
| CPU 推理帧率 | ~0.6 FPS |
| NPU 目标帧率 | 60–200 FPS (待模型转换) |
| 飞控通信波特率 | 500000 baud |
| 控制回路频率 | 10 Hz |
| SD 卡记录目录 | `/media/sdcard/` |
| PYTHONPATH | 项目根目录 (`.`) |

---

*文档生成日期: 2026-07-07*
