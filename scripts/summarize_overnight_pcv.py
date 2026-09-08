#!/usr/bin/env python3
"""Morning report: Q=P·C vs P-only gate vs Tentative P thresholds."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from summarize_q_ablation_cabs_v import (
    _ap05,
    _asr,
    _buffer_totals,
    _gate_mass,
    _pcv_stats,
)


def _load_rows(outdir: Path, prefix: str, ts: str, tag_re: str):
    pat = re.compile(
        rf"{re.escape(prefix)}(clean|remove_early)_({tag_re})_{re.escape(ts)}\.json$"
    )
    rows = []
    for p in sorted(outdir.glob(f"{prefix}*_{ts}.json")):
        m = pat.search(p.name)
        if not m:
            continue
        kind, tag = m.group(1), m.group(2)
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            rows.append({"kind": kind, "tag": tag, "error": str(exc), "path": str(p)})
            continue
        frames = d.get("frames") or []
        qh = float(d.get("q_h") or 0)
        ql = float(d.get("q_l") or 0)
        rows.append(
            {
                "kind": kind,
                "tag": tag,
                "q_h": qh,
                "q_l": ql,
                "gate_q_mode": d.get("gate_q_mode", "pc"),
                "buf_q_mode": d.get("buf_q_mode", "pc_abs"),
                "theta_confirm": d.get("theta_confirm"),
                "n_frames": len(frames),
                "ap05_accept": _ap05(d.get("accepted_metrics")),
                "init_asr": _asr(d, "asr_no_defense"),
                "ours_asr": _asr(d, "asr_defense"),
                "pcv_ego": _pcv_stats(frames, "ego"),
                "gate_mass": _gate_mass(frames, qh, ql),
                "buffer": _buffer_totals(frames),
                "path": str(p),
            }
        )
    return rows


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def _aps(x):
    return "—" if x is None else f"{x:.3f}"


def _metrics_table(rows, extra_cols=None):
    extra_cols = extra_cols or []
    hdr = "| Kind | tag | Q_h | Q_l | " + " | ".join(extra_cols) + (
        " | Init ASR | ours ASR | Accept AP@.5 | frames | conf/atk |"
        if extra_cols
        else "| Init ASR | ours ASR | Accept AP@.5 | frames | conf/atk |"
    )
    if extra_cols:
        hdr = (
            "| Kind | tag | Q_h | Q_l | "
            + " | ".join(extra_cols)
            + " | Init ASR | ours ASR | Accept AP@.5 | frames | conf/atk |"
        )
        sep = "|" + "|".join(["------"] * (8 + len(extra_cols))) + "|"
    else:
        hdr = "| Kind | tag | Q_h | Q_l | Init ASR | ours ASR | Accept AP@.5 | frames | conf/atk |"
        sep = "|------|-----|-----|-----|----------|----------|--------------|--------|---------|"
    lines = [hdr, sep]
    for r in sorted(rows, key=lambda x: (x.get("kind") or "", x.get("tag") or "")):
        if r.get("error"):
            lines.append(f"| {r['kind']} | {r['tag']} |  |  | ERROR {r['error']} |")
            continue
        extras = []
        for c in extra_cols:
            extras.append(str(r.get(c, "—")))
        buf = r.get("buffer") or {}
        ca = f"{buf.get('sum_confirm', 0)}/{buf.get('sum_attack', 0)}"
        mid = (" | ".join(extras) + " | ") if extras else ""
        lines.append(
            f"| {r['kind']} | {r['tag']} | {r['q_h']:.2f} | {r['q_l']:.2f} | {mid}"
            f"{_pct(r['init_asr'])} | {_pct(r['ours_asr'])} | {_aps(r['ap05_accept'])} | "
            f"{r['n_frames']} | {ca} |"
        )
    return lines


def _gate_table(rows, title):
    lines = [
        f"## {title}",
        "",
        "| Kind | tag | mode | N | cut C/A/T | state C/A/T | Q p50 |",
        "|------|-----|------|---|-----------|-------------|-------|",
    ]
    for r in sorted(rows, key=lambda x: (x.get("kind") or "", x.get("tag") or "")):
        g = r.get("gate_mass")
        if not g:
            continue
        lines.append(
            f"| {r['kind']} | {r['tag']} | {r.get('gate_q_mode')} | {g['N']} | "
            f"{_pct(g['certain'])}/{_pct(g['ambiguous'])}/{_pct(g['tentative'])} | "
            f"{_pct(g.get('state_certain'))}/{_pct(g.get('state_ambiguous'))}/"
            f"{_pct(g.get('state_tentative'))} | {g['q_p50']:.3f} |"
        )
    return lines


def _compare_pc_vs_p(pc_rows, p_rows):
    lines = [
        "## P-only vs P·C: does C/V split Ego regions?",
        "",
        "Same (Q_h, Q_l). Δ = P-only − P·C. If |Δ Certain| is tiny, C/V is not doing the partition.",
        "",
        "| Kind | Q_h/Q_l | PC state C/A/T | P state C/A/T | ΔCertain | ΔAP@.5 | Δours ASR |",
        "|------|---------|----------------|---------------|----------|--------|-----------|",
    ]
    idx_p = {(r["kind"], round(r["q_h"], 2), round(r["q_l"], 2)): r for r in p_rows if not r.get("error")}
    for r in sorted(pc_rows, key=lambda x: (x.get("kind") or "", x.get("q_h") or 0, x.get("q_l") or 0)):
        if r.get("error"):
            continue
        key = (r["kind"], round(r["q_h"], 2), round(r["q_l"], 2))
        o = idx_p.get(key)
        if not o:
            continue
        g0, g1 = r.get("gate_mass") or {}, o.get("gate_mass") or {}
        dc = None
        if g0.get("state_certain") is not None and g1.get("state_certain") is not None:
            dc = g1["state_certain"] - g0["state_certain"]
        dap = None
        if r.get("ap05_accept") is not None and o.get("ap05_accept") is not None:
            dap = o["ap05_accept"] - r["ap05_accept"]
        dasr = None
        if r.get("ours_asr") is not None and o.get("ours_asr") is not None:
            dasr = o["ours_asr"] - r["ours_asr"]

        def trip(g):
            return (
                f"{_pct(g.get('state_certain'))}/"
                f"{_pct(g.get('state_ambiguous'))}/"
                f"{_pct(g.get('state_tentative'))}"
            )

        def signed(x, nd=1, pct=True):
            if x is None:
                return "—"
            if pct:
                return f"{100 * x:+.{nd}f}pp"
            return f"{x:+.3f}"

        lines.append(
            f"| {r['kind']} | {r['q_h']:.2f}/{r['q_l']:.2f} | {trip(g0)} | {trip(g1)} | "
            f"{signed(dc)} | {signed(dap, pct=False)} | {signed(dasr)} |"
        )
    return lines


def _verdict(pc_rows, p_rows):
    deltas = []
    for r in pc_rows:
        if r.get("error") or r.get("kind") != "clean":
            continue
        key = (r["kind"], round(r["q_h"], 2), round(r["q_l"], 2))
        idx_p = {
            (x["kind"], round(x["q_h"], 2), round(x["q_l"], 2)): x
            for x in p_rows
            if not x.get("error")
        }
        o = idx_p.get(key)
        if not o:
            continue
        g0 = (r.get("gate_mass") or {}).get("state_certain")
        g1 = (o.get("gate_mass") or {}).get("state_certain")
        if g0 is not None and g1 is not None:
            deltas.append(abs(g1 - g0))
    if not deltas:
        return "Verdict: not enough paired clean runs to compare P vs P·C."
    m = float(np.mean(deltas))
    if m < 0.02:
        return (
            f"Verdict: C/V **does not split** Ego regions in practice "
            f"(mean |Δ Certain| on clean = {100 * m:.2f} pp). Partition is essentially P."
        )
    if m < 0.05:
        return (
            f"Verdict: C/V has **weak** effect on Ego regions "
            f"(mean |Δ Certain| on clean = {100 * m:.2f} pp)."
        )
    return (
        f"Verdict: C/V **does move** Ego region mass "
        f"(mean |Δ Certain| on clean = {100 * m:.2f} pp). Keep Q=P·C if ASR/AP also move."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_ts", required=True, help="Q=P·C ablation timestamp")
    ap.add_argument("--p_ts", required=True, help="P-only gate timestamp")
    ap.add_argument("--tent_ts", required=True, help="Tentative P timestamp")
    ap.add_argument("--outdir", default="logs")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    pc = _load_rows(outdir, "qablate_full_", args.q_ts, r"q\d+_\d+")
    p = _load_rows(outdir, "pgate_full_", args.p_ts, r"p\d+_\d+")
    tent = _load_rows(outdir, "tentp_full_", args.tent_ts, r"c\d+_ql\d+")

    md = [
        "# Overnight: Ego partition P vs C/V, then Tentative P",
        "",
        f"- Q=P·C TS: `{args.q_ts}` (`qablate_full_*`)",
        f"- P-only gate TS: `{args.p_ts}` (`pgate_full_*`, `--gate_q_mode p`)",
        f"- Tentative P TS: `{args.tent_ts}` (`tentp_full_*`, `--buf_q_mode p`)",
        "- Setting: AttFuse, score_thres=0.3, n_ref=200, C=C_abs·V, full val, UAV-only remove_early",
        "",
        _verdict(pc, p),
        "",
        "## Wave 1 — Q=P·C thresholds",
        "",
    ]
    md += _metrics_table(pc)
    md += [""]
    md += _gate_table(pc, "Wave 1 gate mass (Q=P·C)")
    md += ["", "## Wave 2 — gate Q=P only (drop C and V from partition)", ""]
    md += _metrics_table(p)
    md += [""]
    md += _gate_table(p, "Wave 2 gate mass (Q=P)")
    md += [""]
    md += _compare_pc_vs_p(pc, p)
    md += ["", "## Wave 3 — Tentative P (buf_q_mode=p, gate P-only 0.40/0.30 unless tag says otherwise)", ""]
    md += _metrics_table(tent, extra_cols=["theta_confirm", "buf_q_mode"])
    md += [""]
    md += _gate_table(tent, "Wave 3 gate mass (should track q_l)")
    md += ["", "## Ego P/V/C (wave 1 clean, one row per pair)", ""]
    md.append("| Kind | tag | P p50 | V p50 | C p50 | C_abs p50 | V≥0.75 |")
    md.append("|------|-----|-------|-------|-------|-----------|--------|")
    for r in pc + p:
        if r.get("kind") != "clean":
            continue
        pcv = r.get("pcv_ego") or {}
        if not pcv:
            continue

        def g(name):
            return (pcv.get(name) or {}).get("p50")

        def f(x):
            return "—" if x is None else f"{x:.3f}"

        v75 = (pcv.get("V") or {}).get("frac_ge_075")
        md.append(
            f"| {r['kind']} | {r.get('tag')} | {f(g('P'))} | {f(g('V'))} | "
            f"{f(g('C'))} | {f(g('C_abs'))} | {_pct(v75)} |"
        )
    md.append("")
    text = "\n".join(md) + "\n"
    md_path = outdir / "experiment" / f"MORNING_PCV_REPORT_{args.p_ts}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    stable = outdir / "experiment" / "MORNING_PCV_REPORT.md"
    stable.write_text(text, encoding="utf-8")
    js = {
        "q_ts": args.q_ts,
        "p_ts": args.p_ts,
        "tent_ts": args.tent_ts,
        "pc": pc,
        "p": p,
        "tent": tent,
    }
    js_path = outdir / f"overnight_pcv_{args.p_ts}.json"
    js_path.write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    print(text)
    print(f"[saved] {md_path}")
    print(f"[saved] {stable}")
    print(f"[saved] {js_path}")


if __name__ == "__main__":
    main()
