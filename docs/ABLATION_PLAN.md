# 超参消融最终方案（已定）

- 记录日期: 2026-09-08
- 主实验: scene `2025-07-17-16-12_1`, cases `0-29` = 300f
- 攻击: `clean` + `remove_early`
- det NMS: **0.1 主报**, 0.2 附录稳健性
- 指标每格: Accept AP@0.5, P, Ours ASR, ΔASR (vs Init)

---

## 默认锁死

| 参数 | 取值 | 备注 |
|------|------|------|
| \(\theta_{\mathrm{score}}\) | **0.3** | 不扫 |
| \(\theta_{\mathrm{soft}}\) | **0.1**（主默认） | 消融见 G1 |
| \(\kappa_+,\kappa_-\) | 1.4, 1.0 | |
| \(\theta_{\mathrm{confirm}}\) | 0.7 | 消融见 G3 |
| \(\theta_{\mathrm{reject}}\) | **0.1** | 先锁，不扫 |
| \(T_1\) (T_TIMEOUT) | 10 | |
| \(K_{\mathrm{atk}}\) | 10 | |
| \(K\) (POOL_K) | 3 | |
| \(V_{\min}\) | **= UAV 可见性 P5** | **CAL 一次，不扫** |
| \(\beta,\gamma\) | 0.5, 0.5 | |
| \(\tau\) | 0.1 | |

### \(V_{\min}\) 标定（CAL，不定扫）

1. 在 **clean** 上收集 UAV 检出目标的可见性 \(S_{\mathrm{uav}}\)（FOV 内 \(\max(V_{\mathrm{LOS}},C_u)\)，与 pool 一致）
2. \(V_{\min} := P5\)（5% 分位数）
3. 写入实验配置后全程固定；**不做** \(p5\pm0.05\) / 0.25 网格

---

## 五组消融（按 `/` 分组）

### G1 门控门槛 — \(\theta_{\mathrm{soft}},\theta_{\mathrm{score}}\)

- \(\theta_{\mathrm{score}} = 0.3\)（固定）
- \(\theta_{\mathrm{soft}} \in \{0.1,\ 0.15,\ 0.2\}\) → **3 点**

### G2 证据强度 — \(\kappa_+,\kappa_-\)

- \((\kappa_+,\kappa_-) \in \{(1.0,1.0),\ (1.4,1.0),\ (1.4,1.4),\ (2.0,1.0),\ (2.0,1.4)\}\) → **5 点**

### G3 缓冲判决 — \(\theta_{\mathrm{confirm}},\theta_{\mathrm{reject}}\)

- \(\theta_{\mathrm{confirm}} \in \{0.6,\ 0.7,\ 0.8\}\) → **3 点**
- \(\theta_{\mathrm{reject}} = 0.1\)（锁）

### G4 时序寿命 — \(T_1,\ K_{\mathrm{atk}},\ K\)

各一维 3 点（每次只动一个）：

- \(T_1 \in \{5,\ 10,\ 15\}\)
- \(K_{\mathrm{atk}} \in \{5,\ 10,\ 15\}\)
- \(K \in \{1,\ 3,\ 5\}\)

### G5 可见/加权 — \(V_{\min},\beta,\gamma,\tau\)

- \(V_{\min} = P5\)：**不扫**
- \(\beta \in \{0,\ 0.5,\ 1\}\)、\(\gamma \in \{0,\ 0.5,\ 1\}\)、\(\tau \in \{0.05,\ 0.1,\ 0.2\}\)（各一维）

---

## 跑的顺序

1. clean 标定 \(V_{\min}=P5\)
2. **G4 → G1 → G3 → G5 → G2**
3. 主文优先表: **G4 + G1 + G3**；其余附录

---

## 代码对应

| 符号 | 代码 |
|------|------|
| \(\theta_{\mathrm{soft}}\) | `--theta_soft` / `THETA_SOFT` |
| \(\theta_{\mathrm{score}}\) | `--score_thres` / `--theta_p` |
| \(\kappa_+,\kappa_-\) | `KAPPA_POS`, `KAPPA_NEG` |
| \(\theta_{\mathrm{confirm}},\theta_{\mathrm{reject}}\) | `THETA_CONFIRM`, `THETA_REJECT` / CLI |
| \(T_1\) | `T_TIMEOUT` |
| \(K_{\mathrm{atk}}\) | `K_ATK` (`occlusion` / pool) |
| \(K\) | `POOL_K` |
| \(V_{\min}\) | `V_MIN` |
| \(\beta,\gamma,\tau\) | `BETA_V`, `GAMMA_R`, `TAU` |
