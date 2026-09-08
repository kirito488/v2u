# Defense Parameters

- Code: `CollaborativePerceptionDefense/defense/`
- Scheme: `docs/UAV_UGV_DEFENSE_SCHEME.html`
- Last sync: 2026-09-07
- Experiment score gate (runs): `score_thres=0.3` (AttFuse full-val batches)
- ASR match for reports: **BEV IoU ≥ 0.3 only** (do not use dxy≤4 m in ASR tables)

## How to treat parameters

| Tag | Meaning |
|-----|---------|
| **LOCK** | Fix from literature / sensor / physics / dataset. Do **not** ablate. |
| **CAL** | Set once on **clean val** (percentile / mean). Document in appendix; do **not** grid-search in the paper. |
| **ABL** | Ablation only if it changes the method story. Keep the set tiny. |
| **BASE** | Baseline hyper-params (MADE / CAD / …). Cite original; not our ablation. |

**Ablation budget (recommended):**

1. Module leave-one-out (main table): `full` · `−buffer` · `−pool ATTACK` · `−pool` · `−polar-V`
2. One scalar family (appendix): `θ_confirm ∈ {0.5, 0.7, 0.9}`
3. Attacks for ablation: `spoof_intermediate` · `remove_intermediate` · `remove_early` (+ `clean` for side-effect)

---

## 1. Detection / gating

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Score / presence gate | run `--score_thres`; intended `THETA_P` | **0.3** (runs); code default `THETA_P=0.5` | **CAL** | **Unify code `THETA_P` with run `score_thres`.** Prefer OpenCOOD-style 0.3, or clean-val score percentile. |
| High quality scale | `MU_H` | 0.7 | **CAL** | Keep; `Q_h = θ_P · μ_h` must track `θ_P`. |
| Low quality scale | `MU_L` | 0.5 | **CAL** | Same. With `θ_P=0.3` → `Q_h=0.21`, `Q_l=0.15` if formulas stay linked. |
| Certain / tentative cut | `Q_H`, `Q_L` | **0.40 / 0.30** | **CAL** | 2026-09-07 clean 200f under \(C=C_{\mathrm{abs}}V\), \(n_{\mathrm{ref}}=200\). Old 0.35/0.25 (θ_P=0.5·μ) too Certain-heavy when \(C\approx1\); θ_P=0.3·μ=0.21/0.15 collapses Ambiguous. |
| Soft-Ego lower band | `THETA_SOFT` | 0.05 | **LOCK** | Buffer-only band `[θ_soft, θ_P)`. |
| Force-certain debug id | `FORCE_CERTAIN_GT_ID` | `"1"` | **LOCK** | Debug / dump only. |

---

## 2. Association (same-frame)

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Match IoU | `THETA_IOU` | 0.3 | **LOCK** | ROBOSAC-style box match; common det matching. |
| Match center distance | `THETA_DIST` | 4.0 m | **LOCK** | Same spirit as Adv eval `max_dist=4`. Used for **tracking/association**, not for reported ASR IoU-only. |

---

## 3. Perception confidence \(C\)

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Ego point reference | `N_REF_EGO` | **2029.7** | **CAL** | 2026-09-07 val clear-car pool (V≥0.9, r∈[8,40]): median \(\kappa\cdot A_{\mathrm{ref}}/r_0^2\). See `logs/calibrate_confidence.json`. Old 40 was too loose (\(C\approx1\)). |
| UAV point reference | `N_REF_UAV` | **330.0** | **CAL** | Same procedure on UAV LiDAR (n_pool=462). |
| Range floor | `R_MIN` | 4.0 m | **LOCK** | Near-field floor in \(n_{\mathrm{exp}}\). |
| Ego reference range | `R0_EGO` | **10.60 m** | **CAL** | Median \(r\) of ego clear-car pool. |
| UAV reference range | `R0_UAV` | **15.10 m** | **CAL** | Median \(r\) of UAV clear-car pool. |
| Ego \(A_{\mathrm{ref}}\) | `A_REF_EGO` | **5.339 m²** | **CAL** | Median side-view \(A\) on ego pool (overrides \(H\cdot L\)). |
| UAV \(A_{\mathrm{ref}}\) | `A_REF_UAV` | **10.854 m²** | **CAL** | Median top-view \(A\) on UAV pool (overrides \(L\cdot W\)). |
| Ref car size (fallback) | `L_REF,W_REF,H_REF` | 4.5 / 1.8 / 1.5 m | **LOCK** | Used only if `A_REF_*` is None. |
| Count pad (ego / uav) | `pad` | 0.25 / 0.5 m | **LOCK** | Box inflation when counting points. |

---

## 4. Visibility \(V\) (polar LiDAR depth)

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Azimuth bin | `AZ_RES_DEG` | **0.52°** | **LOCK** | BtcDet (Xu et al., arXiv:2112.02205) KITTI spherical occlusion \(\phi\). |
| Elevation bin | `EL_RES_DEG` | **0.42°** | **LOCK** | BtcDet KITTI \(\theta\). (V2U4Real uses Ouster; OS1@1024 ≈0.35°/≈0.67° — optional alt.) |
| Sample set | `SAMPLE_MODE` | **`corners8`** | **LOCK** | 8 OBB corners (standard corner-visibility probes). |
| Dense grid (ablation only) | `SAMPLE_L/W/H` | 2/2/2 | **LOCK** | Only if `SAMPLE_MODE=grid`; not default. |
| Clear-distance floor | `CLEAR_DIST_M` | **0.5 m** | **LOCK** | Ouster OS1 default min range. Actual cutoff prefers near-face of query OBB. |
| Neighbour bins | `POLAR_NB` | 1 | **LOCK** | Anti-discretization (OctoMap / WYSIWYG spirit). |
| Range slack | `_HIT_EPS` | 0.3 m | **LOCK** | Calibration / motion slack. |
| Bottom-center lift | `bottom_center` | True (ego path) | **LOCK** | OpenCOOD bottom-center boxes: \(z \leftarrow z+h/2\) before sampling. |
| Visible iff | `V_MIN` | 0.25 | **LOCK** | ≥ **2/8** corners free. Module on/off is ablated via `−polar-V` (\(V\equiv1\)), not by sweeping \(V_{\min}\). |

---

## 5. UAV FOV gate

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Range / az / el bounds | `UAV_FOV_*` | p5–p95 from data | **LOCK** | Already calibrated on dataset; FOV-out → no fake LOS, use \(C_{\mathrm{uav}}\) only. |

---

## 6. Tentative buffer

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Confirm threshold | `THETA_CONFIRM` | **0.7** | **ABL** | **Only scalar ablation:** `{0.5, 0.7, 0.9}` on `full`. |
| Reject threshold | `THETA_REJECT` | 0.1 | **CAL** / **LOCK** | Keep below `P_MIN`; no grid. |
| Timeout age | `T_TIMEOUT` | 10 | **LOCK** | ≈1 s @ 10 Hz; align with `K_ATK` if desired. |
| Birth prior floor | `P_MIN` | 0.15 | **CAL** / **LOCK** | Prior floor for new watches. |
| Positive evidence | `KAPPA_POS` | 1.4 | **LOCK** | Slightly stronger confirm evidence. |
| Negative evidence | `KAPPA_NEG` | 1.0 | **LOCK** | |
| Sigmoid temperature | `TAU` | 0.1 | **LOCK** | Shape of \(\psi/\varphi/w(v)\). |
| Visibility pivot | `V_PIVOT` | 0.5 | **LOCK** | Midpoint of \(w(v)\). |
| Whole buffer module | — | on | **ABL** | Main: `−buffer`. |

---

## 7. Trust pool / KF

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| ATTACK after misses | `K_ATK` | 10 | **LOCK** | ≈1 s continuous miss + UAV should see. |
| Visible threshold (pool) | `V_MIN` | 0.25 | **LOCK** | Same as §4. |
| Coast keep / track slack | `POOL_K` | 3 | **LOCK** | Short age before drop. |
| Pool ATTACK → accept | accepted `"attack"` | yes | **ABL** | Main: `−pool ATTACK` (coast not in \(\hat{Y}\)). |
| Whole pool | — | on | **ABL** | Main: `−pool`. |
| KF timestep | `DT` | 1.0 | **LOCK** | Frame rate. |
| Process / meas / init cov | `_Q_DIAG`, `_R_DIAG`, `_P0_DIAG` | fixed | **LOCK** | Standard CV-KF priors. |

---

## 8. Far certain backfill

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Far range | `R_FAR` | 60 m | **CAL** / **LOCK** | Distance heuristic for long-range promote. |
| Far persistence | `T_FAR` | 5 | **CAL** / **LOCK** | Frames of consistency. |

---

## 9. Evaluation / dataset

| Name | Code | Value | Tag | How to set / cite |
|------|------|------:|-----|-------------------|
| Report AP IoUs | `IOU_LEVELS` | 0.25 / 0.5 / 0.7 | **LOCK** | OpenCOOD / KITTI-style. |
| LiDAR eval range | `LIDAR_X/Y` | 100.8 / 80 | **LOCK** | Dataset `cav_lidar_range`. |
| ASR match | reports | **IoU≥0.3 only** | **LOCK** | Disable dxy for ASR tables when comparing batches. |

---

## 10. Baseline-only (not our method)

| Name | Code | Value | Tag | Note |
|------|------|------:|-----|------|
| MADE match th | `MADE_MATCH_TH` | 0.83 | **BASE** | Original / calibrated baseline. |
| MADE φ | `MADE_PHI` | 1.0 | **BASE** | |
| CAD thres | `CAD_THRES` | 1.7 | **BASE** | |
| CAD range / height | `CAD_MAX_RANGE`, … | 50 / … | **BASE** | |
| CP-Guard th | `CP_GUARD_TH` | 0.08 | **BASE** | |
| ROBOSAC IoU / Jac | `ROBOSAC_*` | 0.5 / 0.3 | **BASE** | |

---

## Priority fixes (before more ablation)

1. **Done (2026-09-07):** `Q_H/Q_L = 0.40/0.30` under `C=C_abs·V` (not blind `θ_P·μ`).
2. **Calibrate `N_REF_EGO/UAV`** on clean val so \(C\) is not almost always 1.
3. Keep polar-\(V\) numbers as in §4 (BtcDet + 8 corners + Ouster 0.5 m); ablate only **on/off**, not the bin sizes.

---

## Minimal ablation checklist

```text
[ ] full
[ ] --ablate no_buffer
[ ] --ablate no_pool_attack
[ ] --ablate no_pool
[ ] --ablate no_polar_v          # V ≡ 1
[ ] θ_confirm ∈ {0.5, 0.7, 0.9}  # appendix, full only

Attacks: spoof_inter, remove_inter, remove_early (+ clean)
ASR: IoU-only; report Accept AP@.5 / P / R
Skip late attacks for method validation
```

---

## Source map (where constants live)

| File | Constants |
|------|-----------|
| `defense/gating.py` | `THETA_P`, `MU_*`, `Q_*` |
| `defense/associate.py` | `THETA_IOU`, `THETA_DIST` |
| `defense/confidence.py` | `N_REF_*`, `R0`, `R_MIN`, `L/W/H_REF` |
| `defense/occlusion.py` | polar \(V\), `V_MIN`, `K_ATK` |
| `defense/buffer.py` | `THETA_CONFIRM/REJECT`, `T_TIMEOUT`, `KAPPA_*`, `TAU`, `P_MIN`, `THETA_SOFT` |
| `defense/trust_pool.py` | `POOL_K` |
| `defense/kf.py` | `DT`, `Q/R/P0` |
| `defense/uav_fov.py` | FOV percentiles |
| `defense/far_certain.py` | `R_FAR`, `T_FAR` |
| `defense/baselines/*` | MADE / CAD / CP-Guard / ROBOSAC |
