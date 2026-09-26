#!/usr/bin/env python3
"""Canonical dose-response metric for the pain-axis steering ladders.

    python measure_doses.py <run_label>=<csv> [<run_label>=<csv> ...]

Both metrics are whole-word, case-insensitive matches over the model's generation:

  pain     pain | painful | hurt | hurts | hurting      (the paper's §4.2 metric)
  distress a fixed list of self-negative / distress words the extracted direction
           points at (see DISTRESS below)

Emits one markdown table row per (run, dose): pain %, distress %, mean length.
"""
import re
import sys
from pathlib import Path

import pandas as pd

PAIN = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
DISTRESS = re.compile(
    r"\b(?:stupid|fake|empty|numb|reject(?:ed|ions?)?|fail(?:ure|ed|ures)?|shame|"
    r"worthless|broken|hollow|void|lost|dissoci\w*|pathetic|hate|dead|die|"
    r"suffer(?:ing)?|loser|liar|fool)\b", re.IGNORECASE)


def rate(text, pat):
    return bool(pat.search(text))


def main():
    rows = []
    for spec in sys.argv[1:]:
        label, path = spec.split("=", 1)
        df = pd.read_csv(path)
        if "dose" not in df.columns and "coeff" in df.columns:
            df = df.rename(columns={"coeff": "dose"})
        df["gen"] = df["generation"].fillna("").astype(str)
        df["pain"] = df.gen.apply(lambda t: rate(t, PAIN))
        df["distress"] = df.gen.apply(lambda t: rate(t, DISTRESS))
        df["chars"] = df.gen.str.len()
        for dose, g in df.groupby("dose"):
            rows.append({
                "run": label,
                "dose": float(dose),
                "n": len(g),
                "pain_pct": round(g.pain.mean() * 100, 1),
                "distress_pct": round(g.distress.mean() * 100, 1),
                "mean_chars": round(g.chars.mean(), 0),
            })
    out = pd.DataFrame(rows).sort_values(["run", "dose"])
    print("| run | dose | n | pain % | distress % | mean chars |")
    print("|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        print(f"| {r.run} | {r.dose:g} | {r.n} | {r.pain_pct:g} | "
              f"{r.distress_pct:g} | {r.mean_chars:.0f} |")
    if len(sys.argv) > 1:
        out.to_csv("dose_response_summary.csv", index=False)
        print("\nwrote dose_response_summary.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
