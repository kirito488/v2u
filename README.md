# v2u — UAV–UGV collaborative perception defense (monorepo)

GitHub: https://github.com/kirito488/v2u

This repository bundles the pieces needed to **run** the three-source defense
pipeline (code + fusion/attack deps + model checkpoints). The full V2U4Real
**dataset** is not included (too large); download it separately.

## Layout

| Path | Role |
|------|------|
| `defense/`, `scripts/` | Our defense pipeline |
| `AdvCollaborativePerception/` | Attack / `mvp` OpenCOOD wrappers |
| `V2U4Real/` | OpenCOOD + AttFuse/Where2comm/CoAlign ckpts + Ego/UAV PointPillar (epoch40) |
| `logs/` | Local experiment dumps (gitignored) |

## Still needed locally

- Conda env from `V2U4Real/environment.yml` or `AdvCollaborativePerception/environment.yml` (e.g. `v2u4`)
- Dataset under `V2U4Real/v2u4real/{train,val}` (or keep absolute path `/data/hzy/lxt/V2U4Real-main/v2u4real`)

`defense/paths.py` resolves **bundled siblings first**, then the Linux `/data/hzy/lxt/...` paths.

## Quick start

```bash
cd /path/to/v2u
# install deps / activate conda v2u4
python scripts/run_three_source.py --model attfuse --case 0 --score_thres 0.3
```

## What is / is not uploaded

**Included:** defense code, attack `mvp`+scripts, OpenCOOD sources, fusion checkpoints (~25–44MB each), Ego/UAV late PointPillar `net_epoch40.pth` + yaml.

**Excluded:** full `v2u4real/` dataset (~148GB), training tfevents / intermediate epochs, defense `logs/`, MADE AE dumps, Adv sample `data/`.
