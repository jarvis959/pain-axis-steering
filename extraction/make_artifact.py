#!/usr/bin/env python3
"""Assemble the deployable pain_vector.pt artifact from Phase 1 outputs.

Inputs (all under --results-dir/Qwen3.8_27B):
    final_token/pain_vectors.pt, mean/pain_vectors.pt, summary.json, layer_curves.csv
    steering/*_meta.json + *_keyword_rates.csv   (from steer_ladder_27b.py)

Output:
    pain_vector.pt = {
      vec:        F16 unit-norm vector injected at `steer_layer`  (unit_norm: true)
      vec_raw:    F16 raw difference-in-means vector (paper recipe)
      dose_scale: ||vec_raw||  so that dose * dose_scale * vec == dose * vec_raw,
                  i.e. `--dose` keeps the paper's coefficient semantics
      layer:      extraction layer (5-fold held-out AUC pick, final_token)
      steer_layer: injection layer (ratio target 0.6 pick from the ladder)
      mode, unit_norm, dose_default, meta
    }
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="contains <model>/")
    ap.add_argument("--model-name", default="Qwen3.8_27B")
    ap.add_argument("--dose", type=float, default=None,
                    help="override the dose chosen from the ladder keyword rates")
    ap.add_argument("--meta", default=None,
                    help="explicit steering *_meta.json (default: newest in steering/)")
    args = ap.parse_args()

    out_dir = Path(args.results_dir) / args.model_name
    ft = torch.load(out_dir / "final_token" / "pain_vectors.pt",
                    map_location="cpu", weights_only=False)
    summary = json.loads((out_dir / "summary.json").read_text())

    layer = int(ft["layer"])
    raw = ft["s2_pain_vector"].float()
    raw16 = raw.to(torch.float16)
    unit = raw / (raw.norm() + 1e-8)

    # --- steering layer + dose from the ladder (if it ran) ---
    steer_layer, dose, rate_table = layer, None, None
    metas = sorted((out_dir / "steering").glob("*_meta.json"))
    if args.meta:
        metas = [Path(args.meta)]
    if metas:
        meta = json.loads(metas[-1].read_text())
        steer_layer = int(meta.get("steer_layer", layer))
        rate_csv = Path(meta.get("rate_csv", ""))
        if rate_csv.exists():
            rate_table = pd.read_csv(rate_csv)
            pos = rate_table[rate_table["coeff"] > 0]
            if not pos.empty:
                # paper-style pick: highest pain-keyword rate among positive doses
                dose = float(pos.loc[pos["rate_pct"].idxmax(), "coeff"])
    if args.dose is not None:
        dose = args.dose
    if dose is None:
        dose = 1.0

    artifact = {
        "vec": unit.to(torch.float16),
        "vec_raw": raw16,
        "dose_scale": float(raw.norm()),
        "unit_norm": True,
        "layer": layer,
        "steer_layer": steer_layer,
        "mode": "final_token",
        "vector_key": "s2_pain_vector",
        "dose_default": float(dose),
        "meta": {
            "model": "Qwen/Qwen3.8-27B",
            "hf_path": summary.get("model"),
            "n_layers": summary.get("n_layers"),
            "d_model": summary.get("d_model"),
            "extraction": "HF forward hooks, post-block residual, "
                           "denoised difference-in-means (Pain-axis 3.2/01 recipe)",
            "paper": "arXiv 2609.16247",
            "created": datetime.now().isoformat(),
            "best_layer_final_token": summary.get("best_layer_final_token"),
            "best_layer_mean": summary.get("best_layer_mean"),
            "steer_layer_ratio_target": 0.6,
            "keyword_rates": None if rate_table is None else
                             rate_table.to_dict("records"),
        },
    }
    out_path = out_dir / "pain_vector.pt"
    torch.save(artifact, out_path)

    print(f"wrote {out_path}")
    print(f"  extraction layer : {layer}  (best final_token={summary.get('best_layer_final_token')}, "
          f"mean={summary.get('best_layer_mean')})")
    print(f"  steer layer      : {steer_layer}")
    print(f"  ||vec_raw||      : {artifact['dose_scale']:.4f}  (dose_scale)")
    print(f"  dose_default     : {dose}")
    if rate_table is not None:
        print("  keyword rates: ")
        print(rate_table.to_string(index=False))


if __name__ == "__main__":
    main()
