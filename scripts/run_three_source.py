"""Run three-source detection on one V2U4Real case.

Ego / UAV use dedicated late PointPillars; Init uses --model (attfuse/...).

Usage (from this repo root, or any cwd):
    python scripts/run_three_source.py --model attfuse --case 0
    python scripts/run_three_source.py --model attfuse --case 0 --score_thres 0.3 -v
    python scripts/run_three_source.py --model attfuse --case 0 --det_score_thres 0.05
    python scripts/run_three_source.py --model where2comm --case 1 --save out.json

Per-scene checkpoints (resume after crash):
    --save out.json  → writes out_ckpt/scene_*.pkl after each scene
    re-run the same command to skip finished scenes (use --fresh to ignore)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from dataclasses import replace


def _bind_cuda_device_from_argv():
    """Set CUDA_VISIBLE_DEVICES before any torch import."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=int, default=None)
    args, _ = p.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    return args.gpu


_BIND_GPU = _bind_cuda_device_from_argv()

import numpy as np

# repo root on path
_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.paths import (  # noqa: E402
    MODEL_CKPTS,
    V2U4REAL_ROOT,
    EGO_ID,
    UAV_ID,
    EGO_POINTPILLAR_DIR,
    UAV_POINTPILLAR_DIR,
    setup_import_paths,
    train_dir,
    val_dir,
)

setup_import_paths()

warnings.filterwarnings("ignore", message="invalid value encountered in intersection")

from mvp.data.v2u4real_dataset import V2U4RealDataset  # noqa: E402
from mvp.perception.opencood_perception import OpencoodPerception  # noqa: E402

from defense.attack_gt import annotate_gt, nearest_index  # noqa: E402
from defense.apply_attack import (  # noqa: E402
    apply_attack_by_case_spans,
    normalize_targets,
    primary_target,
    replace_init_with_attack,
)
from defense.buffer import (  # noqa: E402
    BETA_V,
    GAMMA_R,
    KAPPA_NEG,
    KAPPA_POS,
    TAU,
    THETA_CONFIRM,
    THETA_REJECT,
    T_TIMEOUT,
)
from defense.occlusion import K_ATK, V_MIN  # noqa: E402
from defense.trust_pool import POOL_K  # noqa: E402
from defense.gating import (  # noqa: E402
    Q_H,
    Q_L,
    TENTATIVE,
    THETA_P,
    spatial_state,
)
from defense.three_source import (  # noqa: E402
    ThreeSourceDetector,
    dump_boxes,
    filter_car_range_gt,
    results_to_dict,
)
from defense.metrics import (  # noqa: E402
    overlap_frame_line,
    print_attack_miss,
    print_defense_metrics,
    print_metrics_table,
    print_source_overlap,
)
from defense.baselines import DEFENSE_NAMES, apply_baseline  # noqa: E402
from defense.baselines.cad import CAD_THRES  # noqa: E402
from defense.baselines.cp_guard import CP_GUARD_TH  # noqa: E402
from defense.baselines.made import MADE_MATCH_TH  # noqa: E402


def default_ckpt_dir(save_path: str | None) -> str | None:
    """logs/foo.json → logs/foo_ckpt/"""
    if not save_path:
        return None
    root, ext = os.path.splitext(os.path.abspath(save_path))
    if ext.lower() != ".json":
        root = os.path.abspath(save_path)
    return root + "_ckpt"


def _scene_pkl_path(ckpt_dir: str, scene_name: str) -> str:
    safe = scene_name.replace("/", "_").replace(" ", "_")
    return os.path.join(ckpt_dir, "scene_{}.pkl".format(safe))


def _ckpt_manifest_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "manifest.json")


def _ckpt_fingerprint(args) -> dict:
    return {
        "model": args.model,
        "mode": args.mode,
        "level": args.level,
        "score_thres": float(args.score_thres),
        "theta_p": float(getattr(args, "theta_p", 0) or 0),
        "theta_soft": float(getattr(args, "theta_soft", 0) or 0),
        "bev_matcher": str(getattr(args, "bev_matcher", "") or ""),
        "bev_tau": float(getattr(args, "bev_tau", 0.5) or 0.5),
        "det_score_thres": float(getattr(args, "det_score_thres", 0.2) or 0.2),
        "defense": args.defense,
        "n_objects": int(args.n_objects),
        "early_remove_mode": args.early_remove_mode,
        "remove_select": str(getattr(args, "remove_select", "ensure") or "ensure"),
        "iters": int(getattr(args, "iters", 0) or 0),
        "also_ego": bool(args.also_ego),
        "q_h": float(getattr(args, "q_h", 0) or 0),
        "q_l": float(getattr(args, "q_l", 0) or 0),
        "gate_q_mode": str(getattr(args, "gate_q_mode", "p") or "p"),
        "theta_confirm": float(getattr(args, "theta_confirm", 0.7) or 0.7),
        "theta_reject": float(getattr(args, "theta_reject", 0.1) or 0.1),
        "t_timeout": int(getattr(args, "t_timeout", 10) or 10),
        "kappa_pos": float(getattr(args, "kappa_pos", 1.4) or 1.4),
        "kappa_neg": float(getattr(args, "kappa_neg", 1.0) or 1.0),
        "k_atk": int(getattr(args, "k_atk", 10) or 10),
        "pool_k": int(getattr(args, "pool_k", 3) or 3),
        "v_min": float(getattr(args, "v_min", 0.25) or 0.25),
        "beta_v": float(getattr(args, "beta_v", 0.5) or 0.5),
        "gamma_r": float(getattr(args, "gamma_r", 0.5) or 0.5),
        "tau": float(getattr(args, "tau", 0.1) or 0.1),
        "buf_q_mode": str(getattr(args, "buf_q_mode", "p") or "p"),
    }


def load_ckpt_manifest(ckpt_dir: str) -> dict:
    p = _ckpt_manifest_path(ckpt_dir)
    if not os.path.isfile(p):
        return {"done": [], "fingerprint": None}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("done", [])
    return data


def save_ckpt_manifest(ckpt_dir: str, done: list, fingerprint: dict) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "done": list(done),
        "fingerprint": fingerprint,
        "updated": __import__("time").strftime("%F %T"),
    }
    with open(_ckpt_manifest_path(ckpt_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_scene_ckpt(ckpt_dir: str, scene_name: str, cases, frames, lidars, fingerprint: dict) -> None:
    """Persist one finished scene so a later --resume can skip recompute."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = _scene_pkl_path(ckpt_dir, scene_name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "scene": scene_name,
                "cases": list(cases),
                "frames": frames,
                "lidars": lidars,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)
    man = load_ckpt_manifest(ckpt_dir)
    done = [s for s in man.get("done") or [] if s != scene_name]
    done.append(scene_name)
    save_ckpt_manifest(ckpt_dir, done, fingerprint)
    print(
        "[ckpt] saved {} ({} frames) → {}".format(scene_name, len(frames), path),
        flush=True,
    )


def load_scene_ckpt(ckpt_dir: str, scene_name: str):
    path = _scene_pkl_path(ckpt_dir, scene_name)
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return list(blob["frames"]), list(blob["lidars"])


def resume_done_scenes(ckpt_dir: str | None, args, planned_scenes) -> list:
    """Return scene names to skip. Empty if no usable checkpoint."""
    if not ckpt_dir or args.fresh or not os.path.isdir(ckpt_dir):
        return []
    man = load_ckpt_manifest(ckpt_dir)
    done = [s for s in (man.get("done") or []) if s in planned_scenes]
    if not done:
        return []
    fp = man.get("fingerprint") or {}
    want = _ckpt_fingerprint(args)
    # Only compare keys the saved ckpt actually recorded (old runs lack q_h etc.).
    mismatch = [k for k in fp if k in want and fp.get(k) != want[k]]
    if mismatch:
        print(
            "[ckpt] fingerprint mismatch on {}; ignore resume ({})".format(
                ckpt_dir, ", ".join(mismatch)
            ),
            flush=True,
        )
        return []
    missing_pkl = [s for s in done if not os.path.isfile(_scene_pkl_path(ckpt_dir, s))]
    if missing_pkl:
        print("[ckpt] missing pkl for {}; drop those from resume".format(missing_pkl), flush=True)
        done = [s for s in done if s not in missing_pkl]
    if done:
        print(
            "[ckpt] resume {} — skip {} scene(s): {}".format(
                ckpt_dir, len(done), done
            ),
            flush=True,
        )
    return done


def parse_case_ids(cases_str, case0, n_cases, n_total):
    """--cases 0-5,8  or  --case 0 --n_cases 6."""
    if cases_str:
        ids = []
        for part in cases_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                ids.extend(range(int(a), int(b) + 1))
            else:
                ids.append(int(part))
    else:
        ids = list(range(int(case0), int(case0) + int(n_cases)))
    out = []
    for i in ids:
        if i < 0 or i >= n_total:
            print("[warn] skip case {} (valid 0..{})".format(i, n_total - 1))
            continue
        if i not in out:
            out.append(i)
    if not out:
        raise SystemExit("no valid case ids")
    return out


def scenes_from_cases(dataset, case_ids):
    """Group selected 10-frame cases by scenario (do NOT expand to full scene).

    V2U4Real multi_frame cases are 10-frame chunks. Same-scene chunks are
    stitched so pool/buffer stay continuous; attack still runs per chunk.
    Scene boundaries still reset in run_case.
    """
    scenes = OrderedDict()
    for ci in case_ids:
        meta = dataset.cases["multi_frame"][ci]
        name = meta["scenario_name"]
        rec = scenes.setdefault(
            name, {"cases": [], "frame_ts": [], "case_frame_spans": []}
        )
        start = len(rec["frame_ts"])
        rec["frame_ts"].extend(list(meta["frame_ids"]))
        rec["cases"].append(ci)
        rec["case_frame_spans"].append((start, len(rec["frame_ts"])))
    return scenes


def all_dataset_scenes(dataset):
    """Every scenario in val, using only 10-frame multi_frame cases (no leftover)."""
    case_by_scene = OrderedDict()
    for ci, meta in enumerate(dataset.cases["multi_frame"]):
        case_by_scene.setdefault(meta["scenario_name"], []).append(ci)
    scenes = OrderedDict()
    for name, cis in case_by_scene.items():
        scenes[name] = scenes_from_cases(dataset, cis)[name]
    return scenes


def load_fusion_model(name: str) -> OpencoodPerception:
    if name not in MODEL_CKPTS:
        raise SystemExit("unknown model {!r}, choose {}".format(name, list(MODEL_CKPTS)))
    print("[load] intermediate fusion ({}) ...".format(name), flush=True)
    perc = OpencoodPerception(
        fusion_method="intermediate",
        model_name=name,
        model_dir=MODEL_CKPTS[name],
        opencood_root=V2U4REAL_ROOT,
        root_dir=train_dir(),
        validate_dir=val_dir(),
    )
    print("[load] intermediate fusion ({}) ready".format(name), flush=True)
    return perc


def load_late_pointpillar(model_dir: str, score_threshold=None) -> OpencoodPerception:
    if not os.path.isdir(model_dir):
        raise SystemExit("late PointPillar dir not found: {}".format(model_dir))
    print("[load] late PointPillar {} ...".format(model_dir), flush=True)
    perc = OpencoodPerception(
        fusion_method="late",
        model_name="pointpillar",
        model_dir=model_dir,
        opencood_root=V2U4REAL_ROOT,
        root_dir=train_dir(),
        validate_dir=val_dir(),
        score_threshold=score_threshold,
    )
    print("[load] late PointPillar {}  only_cav_id={!r}  det_score={}".format(
        model_dir, getattr(perc.dataset, "only_cav_id", None), score_threshold), flush=True)
    return perc


def cuda_empty_cache(tag: str = "") -> None:
    """Best-effort VRAM reclaim between fusion PGD and late PointPillar loads."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if tag:
                print("[cuda] empty_cache ({})".format(tag), flush=True)
    except Exception:
        pass


def ensure_late_perceptions(detector, ego_ckpt: str, uav_ckpt: str, score_threshold=None) -> None:
    """Load ego/uav PointPillars once, after intermediate attack if deferred."""
    if detector.ego_perception is None:
        cuda_empty_cache("before ego PointPillar")
        detector.ego_perception = load_late_pointpillar(
            ego_ckpt, score_threshold=score_threshold
        )
    if detector.uav_perception is None:
        detector.uav_perception = load_late_pointpillar(
            uav_ckpt, score_threshold=score_threshold
        )


def overlay_pool_states(gate_maps, pool) -> None:
    """Birth this frame: certain (keep ambiguous -> certain). Later hit: pool.
    ATTACK coasts: label matching source dets as attack.
    """
    if pool is None:
        return
    for src, mp in pool.source_state_maps().items():
        dest = gate_maps.setdefault(src, {})
        for i, lab in mp.items():
            lab = str(lab)
            if lab == "certain":
                cur = str(dest.get(int(i), "") or "")
                if cur.startswith("ambiguous -> certain") or cur.startswith("tentative -> certain") or cur.startswith("bev -> certain"):
                    continue
                dest[int(i)] = "certain"
            elif lab == "pool":
                dest[int(i)] = "pool"
            elif lab == "attack":
                dest[int(i)] = "attack"


def _dump_pool_attack_on_init(pool, iou_thres: float, gt_boxes=None, gt_ids=None) -> None:
    """Print pool ATTACK coast boxes under [init] so remove detection is visible."""
    if pool is None:
        return
    from defense.geometry import iou_bev
    from defense.trust_pool import ATTACK as POOL_ATTACK

    rows = [
        r
        for r in (getattr(pool, "records", None) or [])
        if str(getattr(r, "action", "")) == POOL_ATTACK and getattr(r, "box", None) is not None
    ]
    if not rows:
        return
    gts = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
    ids = list(gt_ids or [])
    for k, rec in enumerate(rows):
        b = np.asarray(rec.box, dtype=np.float64).reshape(-1)[:7]
        extra = ""
        if gts.ndim == 2 and gts.shape[0] > 0:
            ious = np.array([float(iou_bev(b, g)) for g in gts], dtype=np.float64)
            dxy = np.hypot(gts[:, 0] - b[0], gts[:, 1] - b[1])
            j = int(np.argmax(ious))
            jn = int(np.argmin(dxy))
            use = j if float(ious[j]) > float(ious[jn]) + 1e-6 else jn
            iou = float(ious[use])
            dist = float(dxy[use])
            gid = ids[use] if use < len(ids) else use
            tag = "hit" if iou >= float(iou_thres) else "miss"
            extra = "  {} id={} iou={:.2f} dxy={:.1f}m".format(tag, gid, iou, dist)
        print(
            "  #{:<2d}  B=[{:7.2f} {:7.2f} {:6.2f}  {:5.2f} {:5.2f} {:5.2f} {:6.3f}]  "
            "P=—  C=—  n=—{}  state=attack  (pool coast tid={})".format(
                k,
                b[0], b[1], b[2], b[3], b[4], b[5], b[6],
                extra,
                int(getattr(rec, "tid", -1)),
            )
        )


def fill_dump_states(src, name: str, states, gating=None) -> dict:
    """Gating labels; pool overlay except Certain-into-pool stays certain."""
    out = {}
    scores = np.asarray(getattr(src, "scores", []))
    confs = np.asarray(getattr(src, "confidences", []))
    boxes = np.asarray(getattr(src, "boxes", []))
    n = int(getattr(src, "n", 0) or 0)
    raw = dict(states or {})
    for i in range(n):
        st = raw.get(i)
        if st:
            out[i] = str(st)
            continue
        if gating is not None and i < len(boxes):
            lab = gating.label_for_box(boxes[i])
            if lab:
                out[i] = lab
                continue
        if name == "ego":
            p = float(scores[i]) if i < len(scores) else 0.0
            c = float(confs[i]) if i < len(confs) else 0.0
            out[i] = spatial_state(1, p * c)
        else:
            out[i] = TENTATIVE
    return out


def main():
    ap = argparse.ArgumentParser(description="UAV-UGV three-source detection")
    ap.add_argument(
        "--gpu",
        type=int,
        default=_BIND_GPU,
        help="physical CUDA device id (sets CUDA_VISIBLE_DEVICES before load)",
    )
    ap.add_argument("--model", default="attfuse", choices=list(MODEL_CKPTS),
                    help="Init/fusion intermediate model")
    ap.add_argument(
        "--ego_ckpt",
        default=EGO_POINTPILLAR_DIR,
        help="vehicle-only late PointPillar log dir (only_cav_id=1)",
    )
    ap.add_argument(
        "--uav_ckpt",
        default=UAV_POINTPILLAR_DIR,
        help="UAV-only late PointPillar log dir (only_cav_id=2)",
    )
    ap.add_argument("--case", type=int, default=0, help="first case index")
    ap.add_argument(
        "--n_cases",
        type=int,
        default=1,
        help="number of consecutive 10-frame cases from --case "
             "(not whole scenes). Same-scene cases are stitched for "
             "pool/buffer; attack still runs per 10-frame case",
    )
    ap.add_argument(
        "--cases",
        type=str,
        default=None,
        help="explicit ids, e.g. 0-20 or 0,5,8 (overrides --case/--n_cases)",
    )
    ap.add_argument(
        "--all_scenes",
        action="store_true",
        help="run every scenario in val; reset pool/buffer at each scene",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=None,
        help="max frames from start of the stitched case sequence "
             "(default: all frames from selected cases)",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="do not print per-box dump (still prints metrics)",
    )
    ap.add_argument(
        "--score_thres",
        type=float,
        default=0.3,
        help="UAV/Init D=1 floor (default 0.3); Ego uses --theta_soft",
    )
    ap.add_argument(
        "--theta_p",
        type=float,
        default=None,
        help="Certain / UAV-Init D=1 on P (default: same as --score_thres = 0.3)",
    )
    ap.add_argument(
        "--theta_soft",
        type=float,
        default=0.05,
        help="Ego enter / Ambiguous lower bound (default 0.05)",
    )
    ap.add_argument(
        "--det_score_thres",
        type=float,
        default=0.2,
        help="OpenCOOD NMS score_threshold for late Ego/UAV PointPillar: 0.2 / 0.15 / 0.1 / 0.05",
    )
    ap.add_argument(
        "--n_ref",
        type=float,
        default=None,
        help="Ego n_ref for C_abs (default: confidence.N_REF_EGO)",
    )
    ap.add_argument(
        "--n_ref_uav",
        type=float,
        default=None,
        help="UAV n_ref for C_abs (default: confidence.N_REF_UAV)",
    )
    ap.add_argument(
        "--r0",
        type=float,
        default=None,
        help="Ego reference range r0 for n_exp (default: confidence.R0_EGO)",
    )
    ap.add_argument(
        "--r0_uav",
        type=float,
        default=None,
        help="UAV reference range r0 (default: --r0 or confidence.R0_UAV)",
    )
    ap.add_argument(
        "--a_ref_ego",
        type=float,
        default=None,
        help="Ego A_ref (m^2); default from confidence.area_ref",
    )
    ap.add_argument(
        "--a_ref_uav",
        type=float,
        default=None,
        help="UAV A_ref (m^2); default from confidence.area_ref",
    )
    ap.add_argument(
        "--calib_json",
        type=str,
        default=None,
        help="Apply ego/uav n_ref,r0,A_ref from scripts/calibrate_confidence.py output",
    )
    ap.add_argument(
        "--iou_thres",
        type=float,
        default=0.3,
        help="BEV IoU vs GT for hit (SYSTEM_DESIGN θ_iou=0.3)",
    )
    ap.add_argument(
        "--asr_dist_thres",
        type=float,
        default=0.0,
        help="ASR match distance fallback (m); 0=IoU-only (aligned with nolate reports)",
    )
    ap.add_argument(
        "--q_h",
        type=float,
        default=Q_H,
        help="§5.1 Certain threshold on P (default Q_H={:.2f})".format(Q_H),
    )
    ap.add_argument(
        "--q_l",
        type=float,
        default=Q_L,
        help="§5.1 Ambiguous lower bound on P (default Q_L={:.2f})".format(Q_L),
    )
    ap.add_argument(
        "--gate_q_mode",
        type=str,
        default="p",
        choices=["pc", "p"],
        help="spatial gate score: p=P only (default) or pc=P·C",
    )
    ap.add_argument(
        "--theta_confirm",
        type=float,
        default=None,
        help="Tentative confirm threshold on posterior p (default THETA_CONFIRM=0.7)",
    )
    ap.add_argument(
        "--theta_reject",
        type=float,
        default=None,
        help="Tentative reject threshold (default THETA_REJECT=0.1)",
    )
    ap.add_argument(
        "--t_timeout",
        type=int,
        default=None,
        help="Tentative timeout age T1 (default T_TIMEOUT=10)",
    )
    ap.add_argument(
        "--kappa_pos",
        type=float,
        default=None,
        help="Positive Ego evidence strength κ+ (default KAPPA_POS=1.4)",
    )
    ap.add_argument(
        "--kappa_neg",
        type=float,
        default=None,
        help="Negative Ego miss strength κ- (default KAPPA_NEG=1.0)",
    )
    ap.add_argument(
        "--k_atk",
        type=int,
        default=None,
        help="Attack coast / atk budget K_atk (default K_ATK=10)",
    )
    ap.add_argument(
        "--pool_k",
        type=int,
        default=None,
        help="Pool track max miss K (default POOL_K=3)",
    )
    ap.add_argument(
        "--v_min",
        type=float,
        default=None,
        help="Visibility floor V_min (default V_MIN=0.25)",
    )
    ap.add_argument(
        "--beta_v",
        type=float,
        default=None,
        help="Birth P0 visibility weight β (default BETA_V=0.5)",
    )
    ap.add_argument(
        "--gamma_r",
        type=float,
        default=None,
        help="Birth P0 range weight γ (default GAMMA_R=0.5)",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=None,
        help="ψ / occ_boost softness τ (default TAU=0.1)",
    )
    ap.add_argument(
        "--buf_q_mode",
        type=str,
        default="p",
        choices=["pc_abs", "p"],
        help="buffer ψ score: p=P only (default) or pc_abs=P·C_abs",
    )
    ap.add_argument(
        "--bev_matcher",
        type=str,
        default="",
        help="path to trained BevLocalMatcher ckpt; empty = off",
    )
    ap.add_argument(
        "--bev_tau",
        type=float,
        default=0.5,
        help="promote UAV box to Certain if matcher S >= this (default 0.5)",
    )
    ap.add_argument(
        "--bev_c_min",
        type=float,
        default=0.2,
        help="min UAV C to allow BEV promotion (default 0.2)",
    )
    ap.add_argument(
        "--mode",
        choices=["none", "spoof", "remove", "multi_spoof", "mass_remove"],
        default="none",
        help="none=clean; spoof/remove=single target; "
             "multi_spoof=K ghosts (intermediate); mass_remove=K targets (intermediate)",
    )
    ap.add_argument(
        "--level",
        choices=["early", "intermediate", "late"],
        default="early",
        help="early=改点云; intermediate=BEV PGD 替换 Init; late=naive late 假框/删框",
    )
    ap.add_argument("--iters", type=int, default=20, help="intermediate PGD iterations")
    ap.add_argument(
        "--n_objects",
        type=int,
        default=3,
        help="K for multi_spoof / mass_remove (default 3)",
    )
    ap.add_argument("--also_ego", action="store_true", help="early: also spoof/remove on ego lidar")
    ap.add_argument(
        "--ghost",
        choices=["near", "danger"],
        default="near",
        help="early spoof: near=clone a detected car; danger=blind-spot grid",
    )
    ap.add_argument("--dense", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument(
        "--early_remove_mode",
        choices=["box", "adv"],
        default="box",
        help="early remove: box=UAV overhead OBB delete; adv=Zhang ray+AdvShape",
    )
    ap.add_argument(
        "--remove_select",
        choices=["ensure", "prefer"],
        default="ensure",
        help="early remove target policy: "
             "ensure=≥1 target every frame (UAV-over-ego then fallback); "
             "prefer=any fused det with UAV pts, no ego-score cap, may skip",
    )
    ap.add_argument("--save", type=str, default=None, help="write JSON dump")
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="per-scene checkpoint dir (default: <save>_ckpt when --save is set)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing scene checkpoints (still writes new ones if ckpt_dir set)",
    )
    ap.add_argument(
        "--defense",
        choices=list(DEFENSE_NAMES) + ["all"],
        default="ours",
        help="ours=pool+buffer; none/robosac/cad/made=official baselines; "
             "cp_guard=detection analog only; "
             "all=ours then CPU-replay none/robosac/cad/made",
    )
    ap.add_argument(
        "--jac_thres",
        type=float,
        default=0.3,
        help="ROBOSAC Jaccard threshold (default 0.3)",
    )
    ap.add_argument(
        "--cp_guard_th",
        type=float,
        default=CP_GUARD_TH,
        help="CP-Guard consistency threshold (default {})".format(CP_GUARD_TH),
    )
    ap.add_argument(
        "--cad_thres",
        type=float,
        default=CAD_THRES,
        help="CAD occupancy threshold (default {})".format(CAD_THRES),
    )
    ap.add_argument(
        "--made_th",
        type=float,
        default=MADE_MATCH_TH,
        help="MADE match-loss threshold (default {})".format(MADE_MATCH_TH),
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.gpu is not None:
        print("[cuda] CUDA_VISIBLE_DEVICES={}".format(args.gpu), flush=True)
    elif os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        print("[cuda] CUDA_VISIBLE_DEVICES={}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]), flush=True)
    if args.theta_p is None:
        args.theta_p = args.score_thres
    det_ok = any(abs(float(args.det_score_thres) - x) < 1e-9 for x in (0.2, 0.15, 0.1, 0.05))
    if not det_ok:
        raise SystemExit("--det_score_thres must be 0.2, 0.15, 0.1, or 0.05")
    if args.theta_confirm is None:
        args.theta_confirm = float(THETA_CONFIRM)
    if args.theta_reject is None:
        args.theta_reject = float(THETA_REJECT)
    if args.t_timeout is None:
        args.t_timeout = int(T_TIMEOUT)
    if args.kappa_pos is None:
        args.kappa_pos = float(KAPPA_POS)
    if args.kappa_neg is None:
        args.kappa_neg = float(KAPPA_NEG)
    if args.k_atk is None:
        args.k_atk = int(K_ATK)
    if args.pool_k is None:
        args.pool_k = int(POOL_K)
    if args.v_min is None:
        args.v_min = float(V_MIN)
    if args.beta_v is None:
        args.beta_v = float(BETA_V)
    if args.gamma_r is None:
        args.gamma_r = float(GAMMA_R)
    if args.tau is None:
        args.tau = float(TAU)
    if args.mode in ("multi_spoof", "mass_remove") and args.level != "intermediate":
        print(
            "[warn] mode={!r} forces --level intermediate".format(args.mode),
            flush=True,
        )
        args.level = "intermediate"

    print("[paths] V2U4REAL_ROOT =", V2U4REAL_ROOT)
    print("[paths] val =", val_dir())
    print("[paths] init fusion =", MODEL_CKPTS[args.model])
    print("[paths] ego PointPillar =", args.ego_ckpt)
    print("[paths] uav PointPillar =", args.uav_ckpt)

    dataset = V2U4RealDataset(root_path=val_dir(), mode="val")
    n_total = dataset.case_number("multi_frame")
    if args.all_scenes:
        scenes = all_dataset_scenes(dataset)
        case_ids = [ci for rec in scenes.values() for ci in rec["cases"]]
    else:
        case_ids = parse_case_ids(args.cases, args.case, args.n_cases, n_total)
        scenes = scenes_from_cases(dataset, case_ids)
    quiet = args.quiet or (len(scenes) > 1 and not args.verbose)
    print("[data] cases={} (val has {}) scenes={} ego={} uav={} quiet={}".format(
        case_ids, n_total, list(scenes.keys()), EGO_ID, UAV_ID, quiet), flush=True)

    need_attack = args.mode not in (None, "none", "")
    defer_late = need_attack and args.level in ("intermediate", "late")

    # Optional C calibration (n_ref / r0 / A_ref)
    from defense.confidence import (
        N_REF_EGO,
        N_REF_UAV,
        R0 as _R0_DEF,
        R0_EGO,
        R0_UAV,
        A_REF_EGO,
        A_REF_UAV,
        apply_calibration,
    )

    if args.calib_json:
        with open(args.calib_json, "r", encoding="utf-8") as f:
            calib = json.load(f)
        ego_c = calib.get("ego") or {}
        uav_c = calib.get("uav") or {}
        if ego_c.get("n_ref") is not None:
            args.n_ref = float(ego_c["n_ref"])
        if uav_c.get("n_ref") is not None:
            args.n_ref_uav = float(uav_c["n_ref"])
        if ego_c.get("r0") is not None:
            args.r0 = float(ego_c["r0"])
        if uav_c.get("r0") is not None:
            args.r0_uav = float(uav_c["r0"])
        if ego_c.get("A_ref") is not None:
            args.a_ref_ego = float(ego_c["A_ref"])
        if uav_c.get("A_ref") is not None:
            args.a_ref_uav = float(uav_c["A_ref"])
        print("[calib] loaded", args.calib_json, flush=True)

    if args.n_ref is None:
        args.n_ref = float(N_REF_EGO)
    if args.n_ref_uav is None:
        args.n_ref_uav = float(N_REF_UAV)
    if args.r0 is None:
        args.r0 = float(R0_EGO if R0_EGO is not None else _R0_DEF)
    if args.r0_uav is None:
        args.r0_uav = float(R0_UAV if R0_UAV is not None else args.r0)
    if args.a_ref_ego is None and A_REF_EGO is not None:
        args.a_ref_ego = float(A_REF_EGO)
    if args.a_ref_uav is None and A_REF_UAV is not None:
        args.a_ref_uav = float(A_REF_UAV)
    applied = apply_calibration(
        n_ref_ego=args.n_ref,
        n_ref_uav=args.n_ref_uav,
        r0_ego=args.r0,
        r0_uav=args.r0_uav,
        a_ref_ego=args.a_ref_ego,
        a_ref_uav=args.a_ref_uav,
    )
    print("[calib] C params =", applied, flush=True)

    print("[load] loading fusion model ({}) ...".format(args.model), flush=True)
    perception = load_fusion_model(args.model)
    ego_perception = None
    uav_perception = None
    if not defer_late:
        print("[load] loading 2 late PointPillars ...", flush=True)
        ego_perception = load_late_pointpillar(
            args.ego_ckpt, score_threshold=args.det_score_thres
        )
        uav_perception = load_late_pointpillar(
            args.uav_ckpt, score_threshold=args.det_score_thres
        )
    else:
        print(
            "[load] defer ego/uav PointPillars until after {} attack "
            "(saves VRAM during attack)".format(args.level),
            flush=True,
        )
    bev_matcher = None
    if str(getattr(args, "bev_matcher", "") or "").strip():
        from defense.bev_matcher import load_matcher

        bev_path = os.path.abspath(args.bev_matcher)
        bev_matcher, bev_meta = load_matcher(bev_path)
        print(
            "[load] BEV matcher {}  tau={}  c_min={}  meta={}".format(
                bev_path, args.bev_tau, args.bev_c_min, bev_meta
            ),
            flush=True,
        )
    detector = ThreeSourceDetector(
        perception,
        ego_id=EGO_ID,
        uav_id=UAV_ID,
        n_ref=args.n_ref,
        n_ref_uav=args.n_ref_uav,
        r0=args.r0,
        r0_uav=args.r0_uav,
        score_thres=args.score_thres,
        theta_p=args.theta_p,
        q_h=args.q_h,
        q_l=args.q_l,
        ego_perception=ego_perception,
        uav_perception=uav_perception,
        gate_q_mode=args.gate_q_mode,
        theta_confirm=args.theta_confirm,
        buf_q_mode=args.buf_q_mode,
        theta_soft=args.theta_soft,
        bev_matcher=bev_matcher,
        bev_tau=args.bev_tau,
        bev_c_min=args.bev_c_min,
        theta_reject=args.theta_reject,
        t_timeout=args.t_timeout,
        kappa_pos=args.kappa_pos,
        kappa_neg=args.kappa_neg,
        k_atk=args.k_atk,
        pool_k=args.pool_k,
        v_min=args.v_min,
        beta_v=args.beta_v,
        gamma_r=args.gamma_r,
        tau=args.tau,
    )

    print(
        "score_thres =", args.score_thres,
        "  theta_p =", args.theta_p,
        "  theta_soft =", args.theta_soft,
        "  det_score_thres =", args.det_score_thres,
        "  iou_thres =", args.iou_thres,
        "  n_ref_ego =", args.n_ref,
        "  n_ref_uav =", args.n_ref_uav,
        "  r0 =", args.r0,
        "  r0_uav =", args.r0_uav,
        "  a_ref_ego =", args.a_ref_ego,
        "  a_ref_uav =", args.a_ref_uav,
        "  q_h =", args.q_h,
        "  q_l =", args.q_l,
        "  gate_q_mode =", args.gate_q_mode,
        "  theta_confirm =", args.theta_confirm,
        "  theta_reject =", args.theta_reject,
        "  t_timeout =", args.t_timeout,
        "  kappa_pos =", args.kappa_pos,
        "  kappa_neg =", args.kappa_neg,
        "  k_atk =", args.k_atk,
        "  pool_k =", args.pool_k,
        "  v_min =", args.v_min,
        "  beta_v =", args.beta_v,
        "  gamma_r =", args.gamma_r,
        "  tau =", args.tau,
        "  buf_q_mode =", args.buf_q_mode,
        "  bev_matcher =", args.bev_matcher or "off",
        "  bev_tau =", args.bev_tau,
        "  bev_c_min =", args.bev_c_min,
        "  mode =", args.mode,
        "  level =", args.level,
        "  n_objects =", args.n_objects,
        "  early_remove_mode =", args.early_remove_mode,
        "  remove_select =", args.remove_select,
        "  defense =", args.defense,
    )
    if args.verbose:
        from defense.occlusion import V_MIN, K_ATK
        from defense.buffer import (
            THETA_CONFIRM as _THETA_CONFIRM,
            THETA_REJECT as _THETA_REJECT,
            THETA_SOFT as _THETA_SOFT,
        )
        from defense.trust_pool import POOL_K
        print(
            "[params -v] V_min={}  K_atk={}  θ_confirm={}  θ_reject={}  "
            "θ_soft={}  POOL_K={}  quiet={}".format(
                V_MIN, K_ATK, _THETA_CONFIRM, _THETA_REJECT, _THETA_SOFT, POOL_K, quiet,
            ),
            flush=True,
        )
    ckpt_dir = args.ckpt_dir or default_ckpt_dir(args.save)
    if ckpt_dir:
        print(
            "[ckpt] dir =",
            ckpt_dir,
            ("(fresh)" if args.fresh else "(resume if present)"),
            flush=True,
        )
    skip_scenes = set(resume_done_scenes(ckpt_dir, args, list(scenes.keys())))
    ckpt_fp = _ckpt_fingerprint(args)
    all_frames = []
    all_lidars = []
    for scene_name, rec in scenes.items():
        if scene_name in skip_scenes:
            print(
                "\n===== scene {}  cases {}  SKIP (ckpt) =====".format(
                    scene_name, rec["cases"]
                ),
                flush=True,
            )
            continue
        multi_frame_case = dataset.get_case_by_meta(
            {"scenario_name": scene_name, "frame_ids": rec["frame_ts"]},
            tag="multi_frame",
        )
        frame_ids = list(range(len(multi_frame_case)))
        if args.frames is not None:
            frame_ids = frame_ids[: max(1, args.frames)]
        print("\n===== scene {}  cases {}  frames {}/{}  reset pool/buffer =====".format(
            scene_name, rec["cases"], len(frame_ids), len(multi_frame_case),
        ), flush=True)

        work_case, attack_targets, init_override = apply_attack_by_case_spans(
            args.mode,
            args.level,
            perception,
            dataset,
            multi_frame_case,
            frame_ids,
            rec.get("case_frame_spans") or [(0, len(multi_frame_case))],
            ego_id=EGO_ID,
            att_id=UAV_ID,
            iters=args.iters,
            also_ego=args.also_ego,
            ghost=args.ghost,
            dense=args.dense,
            n_objects=args.n_objects,
            verbose=args.verbose,
            early_remove_mode=args.early_remove_mode,
            remove_select=args.remove_select,
        )
        if defer_late:
            cuda_empty_cache("after {} attack".format(args.level))
            ensure_late_perceptions(
                detector, args.ego_ckpt, args.uav_ckpt,
                score_threshold=args.det_score_thres,
            )
        frames = detector.run_case(work_case, frame_ids)
        if init_override is not None:
            replace_init_with_attack(frames, init_override, detector, work_case)
        for fr, tgt in zip(frames, attack_targets):
            tgts = normalize_targets(tgt)
            fr.attack_targets = tgts if tgts else None
            fr.attack_target = primary_target(tgt)
        # optional single-baseline override (skip ours dump path still has gating)
        if args.defense not in ("ours", "all"):
            for fr, fid in zip(frames, frame_ids):
                lidar = work_case[fid][EGO_ID]["lidar"]
                bl = apply_baseline(
                    args.defense,
                    fr.ego,
                    fr.uav,
                    fr.init,
                    ego_lidar=lidar,
                    jac_thres=args.jac_thres,
                    cp_guard_th=args.cp_guard_th,
                    cad_thres=args.cad_thres,
                    made_th=args.made_th,
                    fusion_z=getattr(fr, "fusion_z", None),
                    made_model=args.model,
                )
                fr.baseline_name = args.defense
                fr.baseline_boxes = None if bl is None else bl.boxes
                fr.baseline_scores = None if bl is None else bl.scores
                fr.baseline_info = None if bl is None else dict(bl.info or {})
        scene_lidars = [work_case[fid][EGO_ID]["lidar"] for fid in frame_ids]
        if ckpt_dir:
            save_scene_ckpt(
                ckpt_dir, scene_name, rec["cases"], frames, scene_lidars, ckpt_fp
            )
        all_frames.extend(frames)
        all_lidars.extend(scene_lidars)
        if not quiet:
            for fr, tgt in zip(frames, attack_targets):
                print("\n" + fr.summary(args.iou_thres))
                ge, ge_ids = filter_car_range_gt(
                    fr.gt_ego, fr.gt_ego_ids, fr.gt_ego_types)
                gu, gu_ids = filter_car_range_gt(
                    fr.gt_uav, fr.gt_uav_ids, fr.gt_uav_types)
                dump_mode = args.mode
                if dump_mode in ("spoof", "multi_spoof"):
                    spoof_on_ego = bool(args.also_ego) and dump_mode == "spoof"
                    ge, ge_ids = annotate_gt(
                        ge, ge_ids, dump_mode, tgt, add_spoof=spoof_on_ego)
                    gu, gu_ids = annotate_gt(
                        gu, gu_ids, dump_mode, tgt, add_spoof=True)
                    dump_eval, dump_eval_ids = annotate_gt(
                        fr.gt_eval, fr.gt_eval_ids, dump_mode, tgt, add_spoof=True)
                    match_eval, match_eval_ids = annotate_gt(
                        fr.gt_eval, fr.gt_eval_ids, dump_mode, tgt, add_spoof=True)
                elif dump_mode in ("remove", "mass_remove"):
                    ge, ge_ids = annotate_gt(ge, ge_ids, dump_mode, tgt)
                    gu, gu_ids = annotate_gt(gu, gu_ids, dump_mode, tgt)
                    dump_eval, dump_eval_ids = annotate_gt(
                        fr.gt_eval, fr.gt_eval_ids, dump_mode, tgt)
                    match_eval, match_eval_ids = dump_eval, dump_eval_ids
                else:
                    dump_eval = np.asarray(fr.gt_eval) if fr.gt_eval is not None else np.zeros((0, 7))
                    dump_eval_ids = list(fr.gt_eval_ids or [])
                    match_eval, match_eval_ids = dump_eval, dump_eval_ids
                for ti, tb in enumerate(normalize_targets(tgt)):
                    tag = "attack-{}".format(
                        dump_mode if ti == 0 else "{}#{}".format(dump_mode, ti + 1)
                    )
                    # Flag remove targets that do not sit on any dump GT (orphan / FP).
                    if dump_mode in ("remove", "mass_remove"):
                        from defense.attack_gt import nearest_index

                        j = nearest_index(dump_eval, tb, max_dist=15.0)
                        if j < 0:
                            tag = tag + "(no-gt)"
                    print(
                        "  [{}] B=[{:7.2f} {:7.2f} {:6.2f}  "
                        "{:5.2f} {:5.2f} {:5.2f} {:6.3f}]".format(
                            tag,
                            float(tb[0]), float(tb[1]), float(tb[2]),
                            float(tb[3]), float(tb[4]), float(tb[5]), float(tb[6]),
                        )
                    )
                print("  [gt-ego Car+range]")
                soft_lo = 0.05
                soft_hi = float(args.score_thres) if args.score_thres is not None else 0.3
                from defense.three_source import soft_ego_gt_extras

                ge_extra = soft_ego_gt_extras(
                    ge,
                    ge_ids,
                    ego_raw=getattr(fr, "ego_raw", None),
                    p_lo=soft_lo,
                    p_hi=soft_hi,
                    iou_thres=float(args.iou_thres),
                    ego_depth=getattr(fr, "ego_depth", None),
                )
                for line in dump_boxes(ge, ge_ids, extras=ge_extra):
                    print(line)
                print("  [gt-uav Car+range]")
                for line in dump_boxes(gu, gu_ids):
                    print(line)
                print("  [gt-eval Car+range]")
                ev_extra = soft_ego_gt_extras(
                    dump_eval,
                    dump_eval_ids,
                    ego_raw=getattr(fr, "ego_raw", None),
                    p_lo=soft_lo,
                    p_hi=soft_hi,
                    iou_thres=float(args.iou_thres),
                    ego_depth=getattr(fr, "ego_depth", None),
                )
                for line in dump_boxes(dump_eval, dump_eval_ids, extras=ev_extra):
                    print(line)
                gate_maps = (
                    fr.gating.source_state_maps()
                    if fr.gating is not None
                    else {"ego": {}, "uav": {}, "init": {}}
                )
                overlay_pool_states(gate_maps, fr.pool)
                match_gt = {
                    "ego": (ge, ge_ids),
                    "uav": (gu, gu_ids),
                    "init": (match_eval, match_eval_ids),
                }
                for name, src in (("ego", fr.ego), ("uav", fr.uav), ("init", fr.init)):
                    print("  [{}]".format(name))
                    states = fill_dump_states(
                        src, name, gate_maps.get(name), gating=fr.gating)
                    gtb, gtid = match_gt[name]
                    lines = list(
                        src.dump_lines(
                            gtb,
                            gtid,
                            iou_thres=args.iou_thres,
                            dist_thres=4.0,
                            gate_states=states,
                        )
                    )
                    printed = False
                    for line in lines:
                        if line.strip() == "(none)":
                            continue
                        print(line)
                        printed = True
                    if name == "init":
                        _dump_pool_attack_on_init(
                            fr.pool,
                            args.iou_thres,
                            gt_boxes=gtb,
                            gt_ids=gtid,
                        )
                        has_atk = any(
                            str(getattr(r, "action", "")) == "attack"
                            for r in (getattr(fr.pool, "records", None) or [])
                        )
                        if not printed and not has_atk:
                            print("  (none)")
                    elif not printed:
                        print("  (none)")
                print(overlap_frame_line(fr, args.iou_thres))
                if fr.gating is not None:
                    print("  [" + fr.gating.summary() + "]")
                    for line in fr.gating.dump_lines():
                        print(line)
                if fr.pool is not None:
                    print("  [" + fr.pool.summary() + "]")
                    for line in fr.pool.dump_lines():
                        print(line)
                if fr.buffer is not None:
                    print("  [" + fr.buffer.summary() + "]")
                    for line in fr.buffer.dump_lines():
                        print(line)
        else:
            for fr in frames:
                print("  " + fr.summary(args.iou_thres))
                print(overlap_frame_line(fr, args.iou_thres))

    if skip_scenes and ckpt_dir:
        pre_frames, pre_lidars = [], []
        for scene_name in scenes:
            if scene_name not in skip_scenes:
                continue
            frs, lids = load_scene_ckpt(ckpt_dir, scene_name)
            pre_frames.extend(frs)
            pre_lidars.extend(lids)
            print(
                "[ckpt] loaded {} ({} frames)".format(scene_name, len(frs)),
                flush=True,
            )
        all_frames = pre_frames + all_frames
        all_lidars = pre_lidars + all_lidars

    if not all_frames:
        raise SystemExit("no frames produced (empty selection / all skipped without ckpt)")

    def _avg_n(attr):
        return float(np.mean([getattr(fr, attr).n for fr in all_frames]))

    print("\navg dets/frame ({} frames, {} cases): ego={:.1f}  uav={:.1f}  init={:.1f}".format(
        len(all_frames), len(case_ids),
        _avg_n("ego"), _avg_n("uav"), _avg_n("init")))

    print_metrics_table(all_frames)
    print_defense_metrics(all_frames)
    attack_miss = print_attack_miss(
        all_frames,
        args.mode,
        iou_thres=args.iou_thres,
        dist_thres=args.asr_dist_thres,
    )
    print_source_overlap(all_frames, iou_thres=args.iou_thres)

    compare_rows = None
    if args.defense == "all":
        compare_rows = _print_defense_compare(all_frames, all_lidars, args)

    if args.save:
        from defense.metrics import evaluate_accepted, evaluate_attack_miss, evaluate_defense

        out = {
            "model": args.model,
            "ego_ckpt": args.ego_ckpt,
            "uav_ckpt": args.uav_ckpt,
            "cases": case_ids,
            "scenes": {n: rec["cases"] for n, rec in scenes.items()},
            "score_thres": args.score_thres,
            "theta_p": args.theta_p,
            "theta_soft": args.theta_soft,
            "bev_matcher": args.bev_matcher,
            "bev_tau": args.bev_tau,
            "bev_c_min": args.bev_c_min,
            "det_score_thres": args.det_score_thres,
            "n_ref": args.n_ref,
            "n_ref_uav": args.n_ref_uav,
            "r0": args.r0,
            "r0_uav": args.r0_uav,
            "a_ref_ego": args.a_ref_ego,
            "a_ref_uav": args.a_ref_uav,
            "calib_json": args.calib_json,
            "q_h": args.q_h,
            "q_l": args.q_l,
            "gate_q_mode": args.gate_q_mode,
            "theta_confirm": args.theta_confirm,
            "theta_reject": args.theta_reject,
            "t_timeout": args.t_timeout,
            "kappa_pos": args.kappa_pos,
            "kappa_neg": args.kappa_neg,
            "k_atk": args.k_atk,
            "pool_k": args.pool_k,
            "v_min": args.v_min,
            "beta_v": args.beta_v,
            "gamma_r": args.gamma_r,
            "tau": args.tau,
            "buf_q_mode": args.buf_q_mode,
            "mode": args.mode,
            "level": args.level,
            "n_objects": args.n_objects,
            "early_remove_mode": args.early_remove_mode,
            "remove_select": args.remove_select,
            "defense": args.defense,
            "defense_metrics": evaluate_defense(all_frames),
            "accepted_metrics": evaluate_accepted(all_frames),
            "attack_miss": attack_miss if isinstance(attack_miss, dict) else evaluate_attack_miss(
                all_frames, args.mode, args.iou_thres, dist_thres=args.asr_dist_thres
            ),
            "compare": compare_rows,
            "frames": results_to_dict(all_frames),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("[save]", os.path.abspath(args.save))


# Official table when --defense all: replay these after ours.
# MADE-only for speed/RAM (skip CAD/ROBOSAC/none occupancy-heavy replay).
COMPARE_BASELINES = ("made",)


def _ap_at(rows, iou=0.5) -> float:
    for r in rows:
        if abs(float(r["iou"]) - float(iou)) < 1e-9:
            return float(r["ap"])
    return float("nan")


def _replay_frames(frames, lidars, name, args):
    out = []
    for fr, lidar in zip(frames, lidars):
        bl = apply_baseline(
            name,
            fr.ego,
            fr.uav,
            fr.init,
            ego_lidar=lidar,
            jac_thres=args.jac_thres,
            cp_guard_th=args.cp_guard_th,
            cad_thres=args.cad_thres,
            made_th=args.made_th,
            fusion_z=getattr(fr, "fusion_z", None),
            made_model=args.model,
        )
        out.append(
            replace(
                fr,
                gating=None,
                buffer=None,
                pool=None,
                baseline_name=name,
                baseline_boxes=None if bl is None else bl.boxes,
                baseline_scores=None if bl is None else bl.scores,
                baseline_info=None if bl is None else dict(bl.info or {}),
            )
        )
    return out


def _print_defense_compare(frames, lidars, args):
    from defense.metrics import evaluate_accepted, evaluate_attack_miss

    print("\n=== Defense comparison (same Ego/UAV/Init dets; replay only) ===")
    print("ours = pool+buffer; none = Init; robosac/made = Init or Ego; cad = lidar occupancy")
    print("MADE = match loss + residual AE if logs/made_ae/{model}_residual_ae.pt exists")
    print("AP/ASR = accepted-state (ours) or baseline output boxes")
    print("{:<10} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        "defense", "AP@.25", "AP@.5", "AP@.7", "ASR_def", "ASR_init"))
    rows = []
    packs = [("ours", frames)]
    for name in COMPARE_BASELINES:
        packs.append((name, _replay_frames(frames, lidars, name, args)))
    for name, frs in packs:
        drows = evaluate_accepted(frs)
        miss = evaluate_attack_miss(
            frs, args.mode, iou_thres=args.iou_thres, dist_thres=args.asr_dist_thres
        )
        rec = {
            "defense": name,
            "ap025": _ap_at(drows, 0.25),
            "ap05": _ap_at(drows, 0.5),
            "ap07": _ap_at(drows, 0.7),
            "asr_defense": miss.get("asr_defense"),
            "asr_no_defense": miss.get("asr_no_defense"),
            "miss_defense": miss.get("miss_defense"),
            "n_attack": miss.get("n_attack"),
        }
        rows.append(rec)
        asr_d = "  n/a" if not rec["n_attack"] else "{:9.1%}".format(rec["asr_defense"])
        asr_i = "  n/a" if not rec["n_attack"] else "{:9.1%}".format(rec["asr_no_defense"])
        print("{:<10} {:>8.3f} {:>8.3f} {:>8.3f} {:>10} {:>10}".format(
            name, rec["ap025"], rec["ap05"], rec["ap07"], asr_d.strip(), asr_i.strip(),
        ))
        if name == "made" and frs:
            inf = getattr(frs[0], "baseline_info", None) or {}
            print("[made] branch={} recon={} th={} match_loss={:.3f} malicious={}".format(
                inf.get("branch"),
                inf.get("recon_loss"),
                inf.get("recon_th"),
                float(inf.get("match_loss") or 0.0),
                inf.get("malicious"),
            ))
    return rows


if __name__ == "__main__":
    main()
