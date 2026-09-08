#!/usr/bin/env python3
"""Summarize full-val Q_h/Q_l ablation + P/V/C distributions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _ap05(metrics):
    if not metrics:
        return None
    for row in metrics:
        if abs(float(row.get("iou", -1)) - 0.5) < 1e-6:
            return float(row.get("ap", row.get("AP", float("nan"))))
    return None


def _asr(d, key):
    am = d.get("attack_miss") or {}
    if isinstance(am, dict) and key in am:
        try:
            return float(am[key])
        except Exception:
            pass
    cmp_ = d.get("compare") or {}
    if key in cmp_:
        try:
            return float(cmp_[key])
        except Exception:
            pass
        return None


def _buffer_totals(frames):
    n_c = n_w = n_a = n_r = n_t = 0
    for fr in frames or []:
        b = fr.get("buffer") or {}
        n_c += int(b.get("n_confirm") or 0)
        n_w += int(b.get("n_watch") or 0)
        n_a += int(b.get("n_attack") or 0)
        n_r += int(b.get("n_reject") or 0)
        n_t += int(b.get("n_timeout") or 0)
    return {
        "sum_confirm": n_c,
        "sum_watch": n_w,
        "sum_attack": n_a,
        "sum_reject": n_r,
        "sum_timeout": n_t,
    }


def _pcv_stats(frames, src="ego"):
    P, C, V, Ca, Q, Qb = [], [], [], [], [], []
    for fr in frames or []:
        s = fr.get(src) or {}
        n = int(s.get("n") or 0)
        scores = s.get("scores") or []
        confs = s.get("confidences") or []
        vs = s.get("visibilities") or []
        cas = s.get("c_abs") or []
        for i in range(min(n, len(scores), len(confs))):
            p = float(scores[i])
            c = float(confs[i])
            v = float(vs[i]) if i < len(vs) else float("nan")
            ca = float(cas[i]) if i < len(cas) else (
                c / v if v == v and v > 1e-6 else c
            )
            P.append(p)
            C.append(c)
            V.append(v)
            Ca.append(ca)
            Q.append(p * c)
            Qb.append(p * ca)
    if not P:
        return None

    def pack(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        qs = np.percentile(a, [25, 50, 75])
        return {
            "N": int(a.size),
            "mean": float(a.mean()),
            "p25": float(qs[0]),
            "p50": float(qs[1]),
            "p75": float(qs[2]),
            "frac_ge_075": float((a >= 0.75).mean()) if a.size else None,
            "frac_ge_1": float((a >= 1.0 - 1e-6).mean()) if a.size else None,
        }

    return {
        "P": pack(P),
        "C": pack(C),
        "V": pack(V),
        "C_abs": pack(Ca),
        "Q": pack(Q),
        "Q_buf": pack(Qb),
    }


def _gate_mass(frames, q_h, q_l):
    """Ego-anchored (d_ego=1) region mass.

    `cut_*` uses Q_ego vs (q_h, q_l) before Dual. `state_*` uses dumped
    post-Dual / far-bypass labels (actual execution).
    """
    qs = []
    st = {"certain": 0, "ambiguous": 0, "tentative": 0, "other": 0}
    for fr in frames or []:
        for o in (fr.get("gating") or {}).get("objects") or []:
            if int(o.get("d_ego") or 0) != 1:
                continue
            qs.append(float(o.get("q_ego") or 0.0))
            lab = str(o.get("state") or "").lower()
            if lab in st:
                st[lab] += 1
            else:
                st["other"] += 1
    if not qs:
        return None
    a = np.asarray(qs, float)
    n = float(a.size)
    return {
        "N": int(a.size),
        "certain": float((a >= q_h).mean()),
        "ambiguous": float(((a >= q_l) & (a < q_h)).mean()),
        "tentative": float((a < q_l).mean()),
        "state_certain": st["certain"] / n,
        "state_ambiguous": st["ambiguous"] / n,
        "state_tentative": st["tentative"] / n,
        "q_mean": float(a.mean()),
        "q_p50": float(np.median(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--outdir", default="logs")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    prefix = "qablate_full_" if args.full else "qablate_"
    pat = re.compile(
        rf"{prefix}(clean|remove_early)_(q\d+_\d+)_{re.escape(args.ts)}\.json$"
    )

    rows = []
    for p in sorted(outdir.glob(f"{prefix}*_{args.ts}.json")):
        m = pat.search(p.name)
        if not m:
            continue
        kind, tag = m.group(1), m.group(2)
        d = json.loads(p.read_text())
        qh = float(d.get("q_h"))
        ql = float(d.get("q_l"))
        frames = d.get("frames") or []
        buf = _buffer_totals(frames)
        acc = d.get("accepted_metrics")
        init_m = None
        # init AP from defense_metrics / compare if present
        for block in (d.get("defense_metrics"), d.get("compare")):
            if isinstance(block, list):
                init_m = block
                break
            if isinstance(block, dict) and "init" in block:
                init_m = block["init"]
                break
        pcv = _pcv_stats(frames, "ego")
        gate = _gate_mass(frames, qh, ql)
        rows.append(
            {
                "kind": kind,
                "tag": tag,
                "q_h": qh,
                "q_l": ql,
                "n_frames": len(frames),
                "ap05_accept": _ap05(acc),
                "init_asr": _asr(d, "asr_no_defense"),
                "ours_asr": _asr(d, "asr_defense"),
                "pcv_ego": pcv,
                "gate_mass": gate,
                "buffer": buf,
                "gate_q_mode": d.get("gate_q_mode", "pc"),
                "path": str(p.relative_to(root)),
            }
        )

    title = "full-val" if args.full else "subset"
    md = [
        f"# Q_h/Q_l ablation ({title}, C=C_abs·V, n_ref=200)",
        "",
        f"- TS: `{args.ts}`",
        "- Modes: clean + remove_early (UAV-only)",
        "- Pairs: 0.35/0.25 (old), 0.40/0.30 (new), 0.42/0.32 (strict), 0.38/0.30",
        "",
        "## Metrics",
        "",
        "| Kind | Q_h | Q_l | Init ASR | ours ASR | Accept AP@.5 | frames |",
        "|------|-----|-----|----------|----------|--------------|--------|",
    ]
    for r in sorted(rows, key=lambda x: (x["kind"], x["q_h"], x["q_l"])):

        def pct(x):
            return "—" if x is None else f"{100 * x:.1f}%"

        def aps(x):
            return "—" if x is None else f"{x:.3f}"

        md.append(
            f"| {r['kind']} | {r['q_h']:.2f} | {r['q_l']:.2f} | "
            f"{pct(r['init_asr'])} | {pct(r['ours_asr'])} | "
            f"{aps(r['ap05_accept'])} | {r['n_frames']} |"
        )

    md += ["", "## Ego P / V / C (detections P≥score_thres in dump)", ""]
    md.append(
        "| Kind | Q_h/Q_l | N | P p50 | V p50 | C p50 | C_abs p50 | Q=P·C p50 | Q_buf p50 | V≥0.75 |"
    )
    md.append(
        "|------|---------|---|-------|-------|-------|-----------|-----------|-----------|--------|"
    )
    for r in sorted(rows, key=lambda x: (x["kind"], x["q_h"], x["q_l"])):
        pcv = r.get("pcv_ego") or {}
        if not pcv:
            continue

        def g(name, key="p50"):
            b = pcv.get(name) or {}
            return b.get(key)

        def f(x, nd=3):
            return "—" if x is None else f"{x:.{nd}f}"

        def pct(x):
            return "—" if x is None else f"{100 * x:.0f}%"

        md.append(
            f"| {r['kind']} | {r['q_h']:.2f}/{r['q_l']:.2f} | "
            f"{(pcv.get('P') or {}).get('N', '—')} | "
            f"{f(g('P'))} | {f(g('V'))} | {f(g('C'))} | {f(g('C_abs'))} | "
            f"{f(g('Q'))} | {f(g('Q_buf'))} | "
            f"{pct((pcv.get('V') or {}).get('frac_ge_075'))} |"
        )

    md += ["", "## Gate mass (d_ego=1): cut on Q_ego vs dumped state after Dual", ""]
    md.append(
        "| Kind | Q_h/Q_l | N | cut C/A/T | state C/A/T | Q p50 |"
    )
    md.append("|------|---------|---|-----------|-------------|-------|")
    for r in sorted(rows, key=lambda x: (x["kind"], x["q_h"], x["q_l"])):
        g = r.get("gate_mass")
        if not g:
            continue

        def pct(x):
            return f"{100 * x:.1f}%"

        md.append(
            f"| {r['kind']} | {r['q_h']:.2f}/{r['q_l']:.2f} | {g['N']} | "
            f"{pct(g['certain'])}/{pct(g['ambiguous'])}/{pct(g['tentative'])} | "
            f"{pct(g.get('state_certain'))}/{pct(g.get('state_ambiguous'))}/"
            f"{pct(g.get('state_tentative'))} | {g['q_p50']:.3f} |"
        )

    md.append("")
    text = "\n".join(md) + "\n"
    tag = "full" if args.full else "sub"
    md_path = outdir / "experiment" / f"q_ablation_cabs_v_{tag}_{args.ts}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    js_path = outdir / f"q_ablation_cabs_v_{tag}_{args.ts}.json"
    js_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(text)
    print(f"[saved] {md_path}")
    print(f"[saved] {js_path}")


if __name__ == "__main__":
    main()
