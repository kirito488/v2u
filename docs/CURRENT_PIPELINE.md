# 当前实现流程（以代码为准）

- 记录日期: 2026-09-09
- 入口: `scripts/run_three_source.py` → `ThreeSourceDetector.run_frame` → `apply_defense`
- 方案稿: `docs/UAV_UGV_DEFENSE_SCHEME.html`（设计意图）
- 本文只写 **仓库里实际在跑的逻辑**。与方案稿 / `docs/params.md` / `docs/ABLATION_PLAN.md` 不一致时，以本文件 + 源码为准。

CLI 默认（不设 `--bev_matcher`、`--mode none`）：

| 项 | 默认 | 代码位置 |
|---|---|---|
| NMS | `--det_score_thres=0.2` | late Ego/UAV PointPillar |
| Ego 进门 | `--theta_soft=0.05` | `gating.THETA_SOFT` |
| UAV/Init / Certain 切 | `--score_thres=θ_P=0.3` | `gating.THETA_P` |
| 门控分 | `gate_q_mode=p`（只看 \(P\)） | `gating.gate_frame` |
| buffer 证据 | `buf_q_mode=p`，`θ_confirm=0.7` | `buffer.py` |
| BEV matcher | 关 | `--bev_matcher` 空 |
| 攻击 | `--mode none` | 不改点云、不换 Init |

消融计划里的 \(\theta_{\mathrm{soft}}=0.1\) **还不是 CLI 默认**。

数据: V2U4 val 的 10 帧 `multi_frame` case。同 scene 拼起来（pool/buffer 连续），scene 边界 `reset`。攻击仍按 10 帧一块做。

---

## 0. 一帧总序

```text
读 case
  →（可选）apply_attack_by_case_spans
  → 三源 det + C
  → P 过滤（Ego θ_soft / UAV·Init θ_score）
  → gate_frame（Certain / Ambiguous / Tentative → Dual 硬抬）
  →（可选）BEV 提升
  → force Certain(GT id=1)、far Certain
  → TrustPool.step
  → leftover = 没被 pool 占走的门控目标
  → TentativeBuffer.step
  → ingest(leftover Certain ∪ CONFIRM)
  → Ŷ
  → AP / ASR
```

跨帧状态只在同一 scene：`run_case` 开头 `tent_buffer / trust_pool / far_tracker.reset()`。

---

## 1. 三源检测

`defense/three_source.py`

| 源 | 模型 | 输入 | 框坐标系 |
|---|---|---|---|
| Ego | late PointPillar（`only_cav_id=1`） | 车端点云 | ego lidar |
| UAV | late PointPillar（`only_cav_id=2`） | UAV 点云（OpenCOOD 内部投到 ego） | ego lidar |
| Init | intermediate AttFuse（`--model`） | 两端点云 | ego lidar |

框: OpenCOOD **底心** 7 维 \([x,y,z,l,w,h,\psi]\)。每源再挂 \(P,C,n,V,C_{\mathrm{abs}}\)。

`--det_score_thres ∈ {0.2, 0.1, 0.05}` 只改 Ego/UAV late NMS，不改门控阈值。

---

## 2. 感知置信 \(C = C_{\mathrm{abs}}\cdot V\)

`defense/confidence.py`，默认 `USE_CN=False`（旧 \(C_n=n/n_{\exp}\) 只留给消融）。

先把底心抬到几何中心 \(z\leftarrow z+h/2\)，再在源雷达框里数点（Ego pad \(0.25\,\mathrm{m}\)，UAV \(0.5\,\mathrm{m}\)）：

\[
C_{\mathrm{abs}}=\mathrm{clip}(n/n_{\mathrm{ref}},\,0,\,1)
\]

代码默认 \(n_{\mathrm{ref}}^{\mathrm{ego}}=40\)、\(n_{\mathrm{ref}}^{\mathrm{uav}}=25\)。`--calib_json` 可覆盖。

\(V\): `build_polar_depth` 做传感器系 \((\mathrm{az},\mathrm{el})\) 最小距离图（\(0.52^\circ/0.42^\circ\)，邻域 \(\pm 1\) bin）。对框采射线（代码默认 `SAMPLE_MODE="grid"`，\(5\times 5\times 3=75\)；文档里写的 8 角不是当前默认）。射线被挡当且仅当该方向已有回波距离小于框近面减去 \(\varepsilon=0.3\,\mathrm{m}\)：

\[
V=\frac{\#\{\text{自由射线}\}}{M}
\]

Ego 路径出 lidar range 则 \(V=0\)。

Init **不是第三台雷达**，是两端平均：

\[
C_{\mathrm{init}}=\tfrac12(C_e+C_u),\quad
V_{\mathrm{init}}=\tfrac12(V_e+V_u),\quad
C_{\mathrm{abs,init}}=\tfrac12(C_{\mathrm{abs},e}+C_{\mathrm{abs},u})
\]

---

## 3. 进门过滤

`run_frame`:

- Ego 留下 \(P\ge\theta_{\mathrm{soft}}\)（CLI 默认 0.05）
- UAV / Init 留下 \(P\ge\theta_{\mathrm{score}}\)（默认 0.3）

过滤前的 Ego 整表留作 `ego_raw`：\(\theta_{\mathrm{soft}}\le P<\theta_P\) 只给 buffer 当 **soft Ego 证据**，不进关联。

---

## 4. 同帧关联

`defense/associate.py`

`hungarian_match`: 代价 \(1-\mathrm{IoU}_{\mathrm{BEV}}\)，只留 \(\mathrm{IoU}\ge\theta_{\mathrm{iou}}=0.3\)。

顺序固定:

1. Ego \(\leftrightarrow\) UAV
2. Ego \(\leftrightarrow\) Init
3. leftover UAV \(\leftrightarrow\) leftover Init

Pool / buffer 跨帧用 `hungarian_match_pool`: \(\mathrm{IoU}\ge 0.3\) **或** 中心距 \(\le\theta_d=4\,\mathrm{m}\)。

\[
D_{\mathrm{src}}=1 \iff \text{该源过了自己的 }P\text{ 门槛，且 Hungarian 配上}
\]

Ego 门槛是 \(\theta_{\mathrm{soft}}\)，UAV/Init 是 \(\theta_P=0.3\)。

---

## 5. 空间门控（§5.1 + §6.3）

`defense/gating.py`，默认 `gate_q_mode="p"`，**状态只看 \(P\)，不看 \(C/V\)**：

\[
Q_{\mathrm{ego}}=
\begin{cases}
P_e & \texttt{gate\_q\_mode=p}\\
P_e\cdot C_e & \texttt{gate\_q\_mode=pc}
\end{cases}
\]

\[
\begin{aligned}
D_e=1,\; Q\ge Q_H &\Rightarrow \textbf{Certain}\\
D_e=1,\; Q_L\le Q< Q_H &\Rightarrow \textbf{Ambiguous}\\
\text{否则} &\Rightarrow \textbf{Tentative}
\end{aligned}
\]

代码: \(Q_H=\theta_P=0.3\)，\(Q_L=\theta_{\mathrm{soft}}\)（默认 0.05）。  
`docs/params.md` 里的 \(0.40/0.30\) 是旧 \(P\cdot C\) 标定，**当前 P-only 实现没用**。

**Dual 硬抬**（`resolve_ambiguous`）:

\[
\mathrm{Dual}=(D_{\mathrm{uav}}=1)\land(D_{\mathrm{init}}=1)
\]

Ambiguous 且 Dual \(\Rightarrow\) Certain（`from_ambiguous=True`）；否则变 Tentative。  
**没有 Ego 的 UAV/Init leftover 一律 Tentative**，Dual 救不了。

框优先 Ego；无 Ego 用 UAV，再没有用 Init。

---

## 6. 局部 BEV 共识（默认关）

`defense/bev_matcher.py`

`--bev_matcher` 非空时，在 `gate_frame` **之后**立刻 `promote_uav_bev_certain`（现亦覆盖 Init 检出框）。

网络: 共享 \(1\times 1\) conv \(\to 64\) 维；UAV crop 做 Q，Ego crop 做 K/V，4-head MHA；conv+GAP+sigmoid 得 \(S\in(0,1)\)。  
crop: BEV 上以框 \((x,y)\) 为中心 \(k=7\)（网格 \(x\in[-100.8,100.8]\)，\(y\in[-80,80]\)，voxel \(0.4\)，stride \(2\)，默认 in_ch \(384\)）。框几何可来自 Ego/UAV/Init，但 **crop 始终取 Ego 与 UAV 两路 BEV**。

提升条件（**不要求 Ego 检出**）:

\[
\text{非 Certain},\;
\big(
  (D_{\mathrm{uav}}=1 \land C_{\mathrm{uav}}\ge c_{\min})
  \;\lor\;
  (D_{\mathrm{init}}=1 \land C_{\mathrm{init}}\ge c_{\min})
\big),\;
S\ge\tau
\;\Rightarrow\; \text{state}=\textbf{Certain},\; \texttt{from\_bev}
\]

CLI 默认 \(\tau=0.5\)，\(c_{\min}=0.2\)。val 前 30 cases（2026-09-09）: \(\tau=0.5\) 的 no-Ego 精度 37.5%；\(\tau=0.7/0.8\) 更干净。建议主实验 \(\tau=0.7\)。

无 Ego 的 BEV Certain **不会**在 `TrustPool.step` 里出生（那里要求 `ego_i>=0`），走 leftover Certain → `ingest`。

---

## 7. 两个旁路（代码在跑）

### 7.1 `force_certain_for_gt_id`

匹配 GT `object_id=1`（自车）无条件 Certain。IoU \(\ge 0.3\) 或中心距 \(\le 4\,\mathrm{m}\)。  
**用了 GT，评测泄漏。** pool 用同一条规则跳过自车出生。论文主表应关掉。

### 7.2 Far Certain（`far_certain.py`）

无 Ego、有 UAV/Init、距离 \(\ge 60\,\mathrm{m}\) 连续 5 帧且 \(\Delta r\ge 0\) \(\Rightarrow\) Certain，打 `from_far`。  
**不进 pool**，只进 dump / \(\hat Y\)。靠近就降回去。

---

## 8. Trust Pool

`defense/trust_pool.py`，在 buffer **之前**。状态 \([x,y,v_x,v_y]\)，恒速 KF，\(\Delta t=1\)（`defense/kf.py`）:

\[
\mathbf{x}_{t|t-1}=F\mathbf{x},\quad
P_{t|t-1}=FPF^\top+Q
\]

1. 全体 `kf_predict`
2. **Certain 且有 Ego 框**: 配上 → UPDATE（`kf_update`）；配不上 → BIRTH
3. 其它 Ego / Init / UAV: HOLD（只把 \(xy\) 吸到检测，不更新协方差）。优先级 Ego \(>\) Init \(>\) UAV
4. 都配不上: `_uav_sight`
   - FOV 内（距离/方位/仰角 p5–p95）: \(V=\max(\mathrm{LOS}_{\mathrm{UAV}},\,C_{\mathrm{uav}})\)
   - FOV 外: 只用 \(C_{\mathrm{uav}}\)（空深度图经常虚高，不当假 LOS）
   - \(V\ge V_{\min}=0.25\) → **ATTACK**，KF coast **进 \(\hat Y\)**；连续 \(K_{\mathrm{atk}}=10\) 帧 → DROP
   - \(V<V_{\min}\): 最多 coast `POOL_K=3` 帧（MISS），再 DROP

被 pool 占走的源框从 leftover 抠掉。  
**不会**因为原始 \(D_e=1\) 就出生；coast 只发生在已经在池里的轨迹上。

`ingest` **只 birth，不 KF-update**（更新只允许 Certain Ego）。

---

## 9. Tentative Buffer

`defense/buffer.py`。只消化 leftover 里的 **Tentative**。已有 watch 若配上本帧 Certain，静默退出，轨迹交给 pool。

### 出生先验 \(P_0\)（永远加权，缺的一侧当 0）

\[
a(P)=\frac{P}{P+\theta_P}\quad(P\le 0\Rightarrow 0)
\]

\[
f_V=1+\beta(0.5-V),\quad
f_r=1+\gamma\frac{r-r_0}{r+r_0},\quad
\beta=\gamma=0.5
\]

\[
P_0=\mathrm{clip}\!\left(\frac{f_V a_e+f_r a_c}{f_V+f_r},\; P_{\min}+\varepsilon,\; P_{\max}-\varepsilon\right)
\]

\(P_{\min}=0.15\)，\(\varepsilon=0.02\)，\(P_{\max}=1\)。\(a_c\) 用 UAV/Init 里较大的 \(P\)。

### 对数几率（只认 Ego）

默认 `buf_q_mode="p"`，证据分 \(Q=P\)（消融可改 \(P\cdot C_{\mathrm{abs}}\)）:

\[
\psi(Q)=\sigma\!\left(\frac{Q-\theta_P}{\tau}\right),\quad
w(v)=1+\sigma\!\left(\frac{0.5-v}{\tau}\right),\quad \tau=0.1
\]

\[
e=
\begin{cases}
\kappa_{+}\,\psi(Q)\,w(v) & \text{Ego hit}\\
-\kappa_{-}\,v & \text{Ego miss}
\end{cases}
\qquad
\kappa_{+}=1.4,\;\kappa_{-}=1.0
\]

\[
\mathrm{logit}(p)\leftarrow\mathrm{logit}(p)+e
\]

\(V\) 用 **Ego** 极坐标深度。soft Ego 只能更新已有 watch，自己不出生。UAV/Init **从不**加正证据。

| 条件 | 动作 |
|---|---|
| \(p\ge\theta_{\mathrm{confirm}}=0.7\) | **CONFIRM** → 进 \(\hat Y\)，下一拍 `ingest` 进 pool |
| \(p\le\theta_{\mathrm{reject}}=0.1\) | **REJECT** |
| \(\mathrm{age}\ge 10\) 且 \(p\le 0.5\) | **TIMEOUT** |
| 连续可见 miss \(\ge K_{\mathrm{atk}}\) | TIMEOUT |
| 可见但未确认 | **ATTACK 只打标签，不进 \(\hat Y\)**，继续 watch |

和 pool 的关键差别: **buffer ATTACK 不发射**；**pool ATTACK 会 KF coast 进接受集**。

---

## 10. 回灌与本帧输出

`_pool_birth_from(leftover, buf)` 只收:

1. leftover **Certain** 且不是 `from_far`、不是自车
2. 本帧 buffer **CONFIRM**

然后 `seed_velocity_from_trajs`: 轨迹 \(\ge 2\) 点、当前 \(|v|\approx 0\) 时，用 \(\Delta x/\Delta t\) 填 \(v_x,v_y\)。

\[
\hat Y \;=\;
\{\text{pool UPDATE/HOLD/BIRTH/ATTACK}\}
\;\cup\;
\{\text{leftover Certain}\}
\;\cup\;
\{\text{buffer CONFIRM}\}
\]

`run_case` 结束会把 later-confirm / far-promote 的历史 backfill 成 dump 标签 `tentative -> certain`。

---

## 11. 评测

`defense/metrics.py`

Accepted 状态:

`certain` · `pool` · `attack` · `ambiguous -> certain` · `tentative -> certain` · `bev -> certain`

外加 **pool ATTACK coast 框**（可能没有对应源框）。同源去重 IoU \(\ge 0.5\)。

| 指标 | 用什么框 |
|---|---|
| Accept AP | 上述 Accepted（IoU 0.25 / 0.5 / 0.7） |
| Defense AP | `buffer.output_boxes`（少 dump overlay） |
| ASR 防御 | Accept |
| ASR 无防御 | Init |

ASR 匹配报告用 **BEV IoU \(\ge 0.3\)**（关掉 dxy）。  
spoof miss = 鬼框还在 Accept；remove miss = 真目标没被接住。

GT: `obj_type==Car` + lidar range（\(\lvert x\rvert\le 100.8\)，\(\lvert y\rvert\le 80\)）。

---

## 12. 和方案叙事不一致、但代码确实在做的

1. Init 不是独立第三传感器。Dual 用它投票，等于「融合头又同意了」。
2. 默认门控是 P-only；\(C,V\) 进 buffer 的 \(w(v)/e_-\) 和 pool 的 `_uav_sight`，不进 Certain/Ambiguous 切分。
3. 「无 Ego 也能 Certain」只有 BEV 路径；门控本身做不到。
4. Path B 只对已在 pool 的 Certain 轨迹 coast，不对原始 Ego FP 出生。
5. `force_certain_for_gt_id` 是评测泄漏。

---

# 攻击逻辑（以代码为准）

入口: `defense/apply_attack.py` → `AdvCollaborativePerception/scripts/attack_v2u4real_online.py` → `OpencoodPerception.attack_intermediate` / `attack_late`。

默认 `--mode none` **不打**。打开后攻击者固定 **UAV**（`att_id=UAV_ID`），受害方 **Ego**。

---

## A. 接到防御上时发生什么

每个 10 帧 case 单独打（Adv 脚本按 `0..9` 索引），再拼回同一 scene: `apply_attack_by_case_spans`。

先用 **干净 Init（AttFuse）** 跑一遍专供选目标: `atk.run_clean(...)`。

返回三样:

| 返回值 | 含义 |
|---|---|
| `work_case` | 可能被改过的点云 case |
| `targets[t]` | 本帧幽灵框 / 删除目标（ASR 用） |
| `init_override` | 被攻击后的融合框；early 为 `None` |

随后:

```text
frames = detector.run_case(work_case)
if init_override:
    replace_init_with_attack(...)   # 换 Init，reset pool/buffer，整段重跑防御
```

因此:

- **early**: 改 UAV（可选 Ego）点云。Ego late / UAV late / Init **都吃改过的云**。
- **intermediate / late**: 点云不动。Ego、UAV 仍是干净检测；**只有 Init 被换成攻击输出**，然后防御重跑。

`multi_spoof` / `mass_remove` 会强制 `level=intermediate`。  
remove 默认 **每帧至少打一个目标**（选不到机强车弱则回退）；只有场景里既无融合框也无 Car GT 时才会整段 clean。

CLI 攻击相关默认: `--level early`，`--iters 20`，`--n_objects 3`，`--also_ego` 关，`--ghost near`，`--early_remove_mode box`。

---

## B. 模式 × 层级

| | early（改点云） | intermediate（UAV BEV PGD） | late（naive 改框） |
|---|---|---|---|
| `spoof` | 默认 `ghost=near`: 克隆一辆车的点，平移后写入 UAV | 前方危险区 1 个幽灵，PGD 抬分 | UAV 检出里硬塞 1 个分=1 的框再融合 |
| `remove` | 默认 `box`: 俯视 OBB 删 UAV 点 + 铺地面 | 跟住 1 个目标，PGD 压分 | 丢掉 UAV 里距目标 \(d_{xy}^2\le 4\) 的框再融合 |
| `multi_spoof` | — | 每帧独立 K 个幽灵（默认 K=3） | — |
| `mass_remove` | — | 每帧独立压 K 个目标（优先 GT 匹配） | — |

主实验路径（消融计划）: `clean` + `remove --level early --early_remove_mode box`。

---

## C. 目标怎么选

early remove 用 `--remove_select` 选策略（2026-09-09）:

| 值 | 行为 |
|---|---|
| `ensure`（默认） | 每帧至少 1 个目标；机强车弱优先，选不到则逐级回退（含 ego>0.35 / GT） |
| `prefer` | **宽松**：融合 \(P\ge0.3\)、距离≥8 m、UAV pts≥15 即可。**不要求**机强车弱，**不筛** Ego-only>0.35。可跳帧；跟踪可沿用；不 force GT |

`ensure` 回退阶梯:

1. 机强车弱（`util>0.05`，UAV 点数≥15，Ego-only ≤0.35）
2. min-ego（允许车更强）
3. 放宽 UAV 点数
4. 任意融合框（\(P\ge0.3\)，距离内）
5. 最近 Car GT
6. 上一帧世界坐标投影（`hold_position`）

`ensure` 下只有场景里既无融合框也无 Car GT 时才会 `None`。日志会打印 `回退选 N`。
`prefer` 下无合格目标则该帧 skip（`攻击 N 帧 / 跳过 M 帧`）。

### early spoof（`ghost=near`，默认）

用第 0 帧干净 Init，按分数从高到低找:

- \(P\ge 0.3\)，距 Ego \(10\sim 75\,\mathrm{m}\)
- Ego 点云落在框内 \(\ge 15\)（避免抠到自车贴地点）

沿该车航向平移 \(+8/+10/+12/-8/+6/-6\,\mathrm{m}\)，新位置须在 \(4\sim 70\,\mathrm{m}\)，且离其它检出 \(\ge 5\,\mathrm{m}\)。找不到就 **最高分框 +8 m**。无 clean / 源点过少 → 回退 `danger` 射线注入（仍每帧有幽灵）。

几何: 第 0 帧从 **Ego 点云**抠源车，warp 到幽灵框，变到世界系；之后每帧投到 **UAV 雷达系并 append**。`--also_ego` 才同时写入 Ego。幽灵框随 Ego 位姿每帧变回 ego 系，作为 ASR target。

`ghost=danger`: `select_attack_positions` 在 Ego 前方 \(x\in[15,35]\) 放一个 \(4.7\times 1.9\times 1.7\) 的框，走 `LidarSpoofEarlyAttacker` 射线注入。

### early / inter / late 单目标 remove

先跑 ego-only / UAV-only（把另一路点云换成远处 1 点）。

`_select_remove_target` + `_plan_remove_targets(..., ensure_every_frame=(remove_select=='ensure'))`:

- `ensure` 优先: 干净融合 \(P\ge 0.3\)，距离 \(\ge 8\,\mathrm{m}\)，UAV 框内点数 \(\ge 15\)，`util=uav_s-ego_s>0.05`，且 Ego-only \(\le 0.35\)；跟丢后按阶梯回退
- `prefer`: 同样 \(P\ge 0.3\) / 距离 / UAV pts≥15，但 **不看 util、不看 Ego-only 分数**；跟丢后 15 m 内 snap 或 `hold_position`；没有 UAV 可见融合框则 skip，不 force GT

### intermediate 单目标 spoof / late spoof

`_select_spoof_ghost`: 前方危险格点，尺寸固定 \(4.7\times 1.9\times 1.7\)，\(z=\) GT 中位数。投到 UAV 系后做 PGD / late 注入。**全程同一幽灵坐标**（每帧都有）。

### mass_remove

每帧独立: 优先 GT 匹配的 top-K；没有则任意融合框；再没有则最近 Car GT。

### multi_spoof

每帧独立: 无 donor 时用默认车尺寸；无危险位时用前方网格。保证每帧 K 个幽灵。

---
## D. 攻击真正改什么

### early remove `box`（默认）

UAV 系下放大脚印（xy ×1.25，pad 2 m / z 1.5 m）删点；删不够再放大一档。删空则留 1 个远处点。再在脚印里铺 **32 个地面点**（固定种子 0）。  
`--also_ego` 才对 Ego 云做同样操作。

`adv`: 先 Zhang 射线 + AdvShape，再补一层框内删点。

### intermediate PGD（`OpencoodPerception.attack_intermediate`）

只扰动 **UAV 那路** scatter 后的 64 通道特征。每个目标一个窗口: 中心 voxel 周围 `feature_size=5` → \(10\times 10\)，\(\|\delta\|_\infty\le 3\)，Adam `lr=0.05`，迭代 `--iters`（默认 20）。

proposal 与目标 3D IoU \(\ge 0.01\) 进 loss；spoof 若对不上，改用 \(d_{xy}<2\,\mathrm{m}\)（再不行 \(<3\,\mathrm{m}\)），只反传最近 32 个 anchor。

\[
\begin{aligned}
\text{spoof: }& L=\sum_j\frac{1}{n_j}\sum_{i\in M_j}\mathrm{IoU}_{ij}\log(1-p_i)
\quad\text{（抬高 }p\text{）}\\
\text{remove: }& L=\sum_j\frac{1}{n_j}\sum_{i\in M_j}(-\mathrm{IoU}_{ij})\log(1-p_i)
\quad\text{（压低 }p\text{）}
\end{aligned}
\]

spoof 附近峰值 \(p\ge 0.35\) 提前停。online 单目标: spoof 把上一帧扰动 \(/2\) 当下帧初值；remove 原样传递。`sync=0`、`init=False`: 不用射线预改点云。

### late

各 CAV 自己出框再拼:

- spoof: UAV 框列表末尾加目标，分数 **1.0**（坐标做了 \(lwh\) 轴序和底心→几何中心）
- remove: 丢掉 UAV 框里 \((x-x_t)^2+(y-y_t)^2\le 4\) 的

然后 NMS + range 过滤，结果只当 Init。

---

## E. 和三源、ASR 的关系

- **early**: UAV late 和 Init 都能看见改过的云；Ego late 默认看不见（除非 `--also_ego`）。这是「协作通道被污染、车端仍干净」。
- **intermediate / late**: Ego、UAV 检测完全干净，只有融合头 Init 被换掉。门控 Dual、Init leftover、buffer 的 collab \(P\) 吃的是假 Init。
- dump 的 `annotate_gt` 只改打印（spoof 追加 `id=spoof`，remove 标 `id(remove)`），**不改评测 GT**。Remove 目标离 GT \(>15\,\mathrm{m}\) 不会造假 GT，会标 `attack-remove(no-gt)`。

---

## 源码地图

| 文件 | 内容 |
|---|---|
| `scripts/run_three_source.py` | CLI、拼 scene、调攻击、评测打印 |
| `defense/three_source.py` | 三源 det、\(C\)、`apply_defense`、`_pool_birth_from` |
| `defense/gating.py` | 关联后 Certain / Ambiguous / Tentative、Dual |
| `defense/bev_matcher.py` | 局部 BEV \(S\)、提升 Certain |
| `defense/trust_pool.py` | KF pool、ATTACK coast |
| `defense/buffer.py` | \(P_0\)、logit、CONFIRM |
| `defense/kf.py` | 恒速 KF |
| `defense/confidence.py` / `occlusion.py` | \(C_{\mathrm{abs}}\)、极坐标 \(V\) |
| `defense/associate.py` | Hungarian |
| `defense/far_certain.py` | 远距旁路 |
| `defense/metrics.py` | Accept / ASR |
| `defense/apply_attack.py` | 模式分发、按 case 切片、`replace_init_with_attack` |
| `defense/attack_gt.py` | dump 标注（不影响评测 GT） |
| `AdvCollaborativePerception/scripts/attack_v2u4real_online.py` | 选目标、early 改点云、调 PGD |
| `mvp/perception/opencood_perception.py` | `attack_intermediate` / `attack_late` |
| `mvp/attack/lidar_*_attacker.py` | inter spoof/remove、multi spoof |
