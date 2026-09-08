#!/usr/bin/env python3
"""Summarize P / C / V / Q from a three-source JSON dump (C=C_abs*V)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(a, name):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        print(f"  {name}: (empty)")
        return
    qs = np.percentile(a, [10, 25, 50, 75, 90])
    print(
        f"  {name}: N={a.size:5d}  mean={a.mean():.4f}  "
        f"p10={qs[0]:.4f} p25={qs[1]:.4f} p50={qs[2]:.4f} "
        f"p75={qs[3]:.4f} p90={qs[4]:.4f}  "
        f"frac0={(a<=1e-6).mean():.3f} frac1={(a>=1-1e-6).mean():.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    args = ap.parse_args()
    d = json.loads(Path(args.dump).read_text())
    frames = d.get("frames") or []
    print(f"dump={args.dump}  frames={len(frames)}")
    print(
        f"n_ref={d.get('n_ref')} n_ref_uav={d.get('n_ref_uav')} "
        f"q_h={d.get('q_h')} q_l={d.get('q_l')} score_thres={d.get('score_thres')}"
    )
    for src in ("ego", "uav", "init"):
        P, C, V, Q = [], [], [], []
        for fr in frames:
            s = fr.get(src) or {}
            ps = s.get("scores") or []
            cs = s.get("confidences") or []
            vs = s.get("visibilities") or []
            qs = s.get("qualities") or []
            n = min(len(ps), len(cs))
            for i in range(n):
                P.append(float(ps[i]))
                C.append(float(cs[i]))
                if i < len(vs):
                    V.append(float(vs[i]))
                if i < len(qs):
                    Q.append(float(qs[i]))
                else:
                    Q.append(float(ps[i]) * float(cs[i]))
        print(f"\n=== {src} ===")
        _stats(P, "P")
        _stats(C, "C")
        _stats(V, "V")
        _stats(Q, "Q=P*C")
        if P and V and len(P) == len(V):
            corr = float(np.corrcoef(P, V)[0, 1]) if np.std(P) > 0 and np.std(V) > 0 else float("nan")
            print(f"  corr(P,V)={corr:.4f}")
        if C and V and len(C) == len(V):
            # C should track C_abs*V; show V bins of Q
            V = np.asarray(V)
            Q = np.asarray(Q)
            P = np.asarray(P)
            C = np.asarray(C)
            print("  by V coarse → mean P / C / Q:")
            for a, b, lab in [
                (0.0, 0.25, "[0,0.25)"),
                (0.25, 0.5, "[0.25,0.5)"),
                (0.5, 0.75, "[0.5,0.75)"),
                (0.75, 1.01, "[0.75,1]"),
            ]:
                m = (V >= a) & (V < b)
                if not m.any():
                    continue
                print(
                    f"    {lab}: N={m.sum():4d}  "
                    f"P={P[m].mean():.3f}  C={C[m].mean():.3f}  Q={Q[m].mean():.3f}"
                )


if __name__ == "__main__":
    main()
