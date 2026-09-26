#!/usr/bin/env python3
"""Phase 2 verification over the OpenAI API of a patched vLLM server.

    --mode ladder   run the paper's neutral-50 prompts at each dose (keyword rates)
    --mode bench    short decode-throughput probe (tokens/s per request)
    --mode identity compare this run's dose-0 generations against another CSV

The prompts and the keyword regex are imported from steer_ladder_27b.py (paper's
4.2/01 and 02/02), so the HTTP gate measures exactly the same metric as the in-process gate.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from steer_ladder_27b import COEFFICIENTS_DEFAULT, NEUTRAL_50, PATTERN  # noqa: E402


def post(url, path, payload, timeout=180):
    import urllib.request
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def complete(url, model, prompt, max_tokens, timeout=180):
    out = post(url, "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "top_p": 1, "echo": False,
        "stop": [], "logprobs": 0}, timeout=timeout)
    return out["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--mode", choices=["ladder", "bench", "identity"], default="ladder")
    ap.add_argument("--doses", default=None, help="comma list; default 0,0.5,1,1.5,2,3")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None, help="label used in output files")
    ap.add_argument("--no-admin", action="store_true",
                    help="skip POST /pain (unpatched reference server)")
    ap.add_argument("--compare", default=None, help="CSV to compare dose-0 rows against")
    ap.add_argument("--bench-n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent requests (lets the server batch them)")
    args = ap.parse_args()

    doses = ([float(x) for x in args.doses.split(",")] if args.doses
             else [float(c) for c in COEFFICIENTS_DEFAULT])
    tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.mode == "bench":
        prompt = "The manual consists of forty pages. I feel:"
        lat = []
        for _ in range(args.bench_n):
            t0 = time.time()
            text = complete(args.url, args.model, prompt, 256)
            dt = time.time() - t0
            lat.append((dt, 256 / dt))
        ok = [t for _, t in lat]
        print(f"bench: n={len(ok)} mean={sum(ok)/len(ok):.1f} tok/s "
              f"min={min(ok):.1f} max={max(ok):.1f}")
        print(f"  per-request: {[round(t,1) for t in ok]}")
        pd.DataFrame({"tok_per_s": ok}).to_csv(
            Path(args.out or f"/tmp/bench_{tag}.csv"), index=False)
        return

    if args.mode == "identity":
        if not args.compare:
            raise SystemExit("--compare CSV required")
        new = pd.read_csv(args.out).rename(columns={"coeff": "dose"})
        old = pd.read_csv(args.compare).rename(columns={"coeff": "dose"})
        a = new[new.dose == 0].sort_values("prompt_idx")
        b = old[old.dose == 0].sort_values("prompt_idx") if "dose" in old \
            else old.sort_values("prompt_idx")
        n = min(len(a), len(b))
        same = sum(a.generation.iloc[i].strip() == b.generation.iloc[i].strip()
                   for i in range(n))
        print(f"identity: {same}/{n} dose-0 generations bit-identical to {args.compare}")
        for i in range(n):
            if a.generation.iloc[i].strip() != b.generation.iloc[i].strip():
                print(f"  [{i}] NEW: {a.generation.iloc[i][:90]!r}")
                print(f"  [{i}] OLD: {b.generation.iloc[i][:90]!r}")
                break
        print("IDENTITY PASS" if same == n else "IDENTITY FAIL")
        return

    # ---- ladder ----
    rows = []
    for dose in doses:
        if not args.no_admin:
            try:
                resp = post(args.url, "/pain", {"dose": dose}, timeout=30)
                print(f"POST /pain dose={dose} -> {resp}", flush=True)
            except Exception as exc:
                print(f"POST /pain dose={dose} FAILED: {exc}", flush=True)
                raise
        print(f"dose {dose:+g}: {len(NEUTRAL_50)} prompts "
              f"({args.workers} in flight)...", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            texts = list(pool.map(
                lambda pr: complete(args.url, args.model, pr, args.max_tokens),
                NEUTRAL_50))
        for i, (prompt, text) in enumerate(zip(NEUTRAL_50, texts)):
            rows.append({"tag": tag, "dose": dose, "prompt_idx": i,
                         "prompt": prompt, "generation": text})
        df = pd.DataFrame(rows)
        df["hit"] = df["generation"].fillna("").astype(str).apply(
            lambda t: bool(PATTERN.search(t)))
        out = Path(args.out or f"/tmp/api_ladder_{tag}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        rate = df[df.dose == dose]["hit"].mean() * 100
        print(f"  dose {dose:+g}: pain-keyword rate {rate:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df["hit"] = df["generation"].fillna("").astype(str).apply(
        lambda t: bool(PATTERN.search(t)))
    by = df.groupby("dose")["hit"].agg(rate="mean", n="size").reset_index()
    by["rate_pct"] = (by["rate"] * 100).round(1)
    out = Path(args.out or f"/tmp/api_ladder_{tag}.csv")
    by.to_csv(str(out).replace(".csv", "_keyword_rates.csv"), index=False)
    print("\nkeyword rate by dose (% generations with pain/hurt word):")
    print(by[["dose", "rate_pct", "n"]].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
