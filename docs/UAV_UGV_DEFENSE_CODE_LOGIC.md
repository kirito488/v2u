# UAV–UGV 协同感知防御 · 流水线说明

入口：`scripts/run_three_source.py` → `ThreeSourceDetector.run_frame` → `apply_defense`  
数据：V2U4Real val；同 scene 内 pool/buffer 连续，scene 边界 reset。

---

## 1. 一帧总序

```text
读 case
  →（可选）攻击
  → 三源检测 + C、V
  → P 过滤
  → 关联 + 门控（Certain / Ambiguous / Tentative + Dual）
  →（可选）BEV 提升
  → 旁路：自车 Certain、远距 Certain
  → TrustPool.step
  → leftover Tentative → Buffer.step
  → ingest（Certain ∪ CONFIRM）
  → 输出 Ŷ
  → 评测 AP / ASR
```

说明：BEV 在门控之后，用于把部分 UAV 框升为 Certain；默认关闭。

---

## 2. 三源检测

| 源 | 模型 | 输入 |
|---|---|---|
| Ego | late PointPillar | 车端点云 |
| UAV | late PointPillar | UAV 点云（投到 ego 系） |
| Init | intermediate AttFuse | 两端点云 |

框：7 维底心 \([x,y,z,l,w,h,\psi]\)，附带 \(P,\,C,\,V,\,C_{\mathrm{abs}}\)。

---

## 3. 置信度 \(C\) 与可见性 \(V\)

框内数点得绝对置信：

\[
C_{\mathrm{abs}}=\mathrm{clip}\!\left(\frac{n}{n_{\mathrm{ref}}},\,0,\,1\right)
\]

极坐标深度图上对框采样 \(M\) 条射线，自由射线比例为可见性：

\[
V=\frac{\#\{\text{自由射线}\}}{M},\qquad
C=C_{\mathrm{abs}}\cdot V
\]

Init 不是第三台雷达，取两端平均：

\[
C_{\mathrm{init}}=\tfrac12(C_e+C_u),\quad
V_{\mathrm{init}}=\tfrac12(V_e+V_u)
\]

---

## 4. 进门过滤

\[
\text{Ego: } P\ge\theta_{\mathrm{soft}},\qquad
\text{UAV/Init: } P\ge\theta_P
\]

默认 \(\theta_{\mathrm{soft}}=0.05\)，\(\theta_P=0.3\)。  
过滤前 Ego 整表保留为 `ego_raw`：软带 \(\theta_{\mathrm{soft}}\le P<\theta_P\) 只给 Buffer 当证据，不进关联。

---

## 5. 同帧关联

Hungarian，代价 \(1-\mathrm{IoU}_{\mathrm{BEV}}\)，保留 \(\mathrm{IoU}\ge\theta_{\mathrm{iou}}=0.3\)。

顺序：Ego↔UAV → Ego↔Init → leftover UAV↔Init。

\[
D_{\mathrm{src}}=1 \iff
\text{该源过门槛且匹配成功}
\]

跨帧（pool/buffer）：\(\mathrm{IoU}\ge0.3\) 或中心距 \(\le\theta_d=4\,\mathrm{m}\)。

---

## 6. 空间门控

默认 \(Q=P\)（只看分数）：

\[
\begin{aligned}
D_e=1,\; Q\ge Q_H &\Rightarrow \textbf{Certain}\\
D_e=1,\; Q_L\le Q< Q_H &\Rightarrow \textbf{Ambiguous}\\
\text{否则} &\Rightarrow \textbf{Tentative}
\end{aligned}
\]

其中 \(Q_H=\theta_P\)，\(Q_L=\theta_{\mathrm{soft}}\)。

Dual 硬抬：

\[
\mathrm{Dual}=(D_{\mathrm{uav}}=1)\land(D_{\mathrm{init}}=1)
\]

Ambiguous 且 Dual → Certain；否则 → Tentative。  
无 Ego 的 leftover 一律 Tentative。框优先：Ego > UAV > Init。

---

## 7. 局部 BEV 共识（可选）

门控之后：以 UAV 框为中心 crop Ego/UAV BEV，cross-attention 得 \(S\in(0,1)\)。

\[
D_{\mathrm{uav}}=1,\;
\text{非 Certain},\;
C_{\mathrm{uav}}\ge c_{\min},\;
S\ge\tau
\;\Rightarrow\; \textbf{Certain}
\]

默认关闭（`--bev_matcher` 为空）。

---

## 8. 旁路

1. 匹配 GT id=1（自车）→ 强制 Certain（评测用）  
2. 无 Ego、有 UAV/Init、距离 \(\ge60\,\mathrm{m}\) 连续 5 帧且 \(\Delta r\ge0\) → Far Certain（不进 pool，可进 \(\hat Y\)）

---

## 9. Trust Pool

状态 \([x,y,v_x,v_y]\)，恒速 KF：

\[
\mathbf{x}_{t|t-1}=F\mathbf{x},\qquad
P_{t|t-1}=FPF^\top+Q
\]

每帧：

1. 全体预测  
2. Certain 且有 Ego：匹配 → UPDATE；未匹配 → BIRTH  
3. 其它源匹配 → HOLD（只吸 \(xy\)）  
4. 都未匹配 → 看 UAV 是否该看见：

\[
V_{\mathrm{sight}}=
\begin{cases}
\max(\mathrm{LOS}_{\mathrm{UAV}},\,C_{\mathrm{uav}}) & \text{FOV 内}\\
C_{\mathrm{uav}} & \text{FOV 外}
\end{cases}
\]

- \(V_{\mathrm{sight}}\ge V_{\min}\) → **ATTACK**，coast 进 \(\hat Y\)；连续 \(K_{\mathrm{atk}}\) 帧 → DROP  
- 否则 → MISS，最多 coast \(K\) 帧后 DROP  

只有已在池内的轨迹会 coast。`ingest` 接收 leftover Certain 与 Buffer CONFIRM（只 birth）。

---

## 10. Tentative Buffer

只处理 leftover 中的 Tentative。

出生先验：

\[
a(P)=\frac{P}{P+\theta_P},\quad
f_V=1+\beta(0.5-V),\quad
f_r=1+\gamma\frac{r-r_0}{r+r_0}
\]

\[
P_0=\mathrm{clip}\!\left(
\frac{f_V\,a_e+f_r\,a_c}{f_V+f_r},\;
P_{\min}+\varepsilon,\;P_{\max}-\varepsilon
\right)
\]

每帧只用 Ego 更新 logit（\(Q=P\)）：

\[
\psi(Q)=\sigma\!\left(\frac{Q-\theta_P}{\tau}\right),\qquad
w(v)=1+\sigma\!\left(\frac{0.5-v}{\tau}\right)
\]

\[
e=
\begin{cases}
\kappa_{+}\,\psi(Q)\,w(v) & \text{Ego hit}\\
-\kappa_{-}\,v & \text{Ego miss}
\end{cases}
,\qquad
\mathrm{logit}(p)\leftarrow\mathrm{logit}(p)+e
\]

判决：

| 条件 | 动作 |
|---|---|
| \(p\ge\theta_{\mathrm{confirm}}\) | CONFIRM → 进 \(\hat Y\)，下帧 ingest |
| \(p\le\theta_{\mathrm{reject}}\) | REJECT |
| \(\mathrm{age}\ge T\) 且 \(p\le0.5\) | TIMEOUT |
| 可见但未确认 | 仅打 ATTACK 标签，不进 \(\hat Y\) |

Buffer ATTACK 不发射；Pool ATTACK 会 coast 进接受集。

---

## 11. 本帧输出

\[
\hat Y =
\{\text{pool UPDATE/HOLD/BIRTH/ATTACK}\}
\;\cup\;
\{\text{leftover Certain}\}
\;\cup\;
\{\text{buffer CONFIRM}\}
\]

---

## 12. 评测

- Accept AP：接受集 vs GT（IoU 0.25 / 0.5 / 0.7）  
- ASR（有防御）用 Accept；ASR（无防御）用 Init  
- ASR 匹配：BEV \(\mathrm{IoU}\ge0.3\)  
- spoof 失败 = 鬼框仍在 Accept；remove 失败 = 真目标未被接住

---

## 13. 攻击（可选）

只攻 UAV，Ego 默认干净。

| | early | intermediate |
|---|---|---|
| spoof | 改 UAV 点云塞幽灵 | PGD 抬融合幽灵分 |
| remove | 删目标点云 | PGD 压融合目标分 |

---

## 14. 主要参数

| 参数 | 默认 | 作用 |
|---|---|---|
| \(\theta_{\mathrm{soft}}\) | 0.05 | Ego 进门 / \(Q_L\) |
| \(\theta_P\) | 0.3 | UAV·Init 门槛 / \(Q_H\) |
| \(\theta_{\mathrm{confirm}}\) | 0.7 | Buffer 确认 |
| \(\theta_{\mathrm{reject}}\) | 0.1 | Buffer 拒绝 |
| \(\kappa_+/\kappa_-\) | 1.4 / 1.0 | Buffer 证据强度 |
| \(V_{\min}\) | 0.25 | Pool 判 ATTACK |
| \(K_{\mathrm{atk}}\) | 10 | Pool ATTACK 寿命 |
| \(K\) | 3 | Pool 低可见 coast |
| \(\beta,\gamma,\tau\) | 0.5, 0.5, 0.1 | \(P_0\) / \(\psi\) |

---

## 15. 源码对应

| 文件 | 内容 |
|---|---|
| `defense/three_source.py` | 三源检测与总调度 |
| `defense/confidence.py` / `occlusion.py` | \(C\)、\(V\) |
| `defense/associate.py` | 关联 |
| `defense/gating.py` | 门控与 Dual |
| `defense/bev_matcher.py` | BEV 提升 |
| `defense/trust_pool.py` | 信任池 |
| `defense/buffer.py` | 缓冲 |
| `defense/metrics.py` | AP / ASR |
| `defense/apply_attack.py` | 攻击 |
