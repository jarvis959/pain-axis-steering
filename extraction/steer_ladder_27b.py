#!/usr/bin/env python3
"""Steering ladder for Qwen3.8-27B: §4.2 sanity gate.

Port of Pain-axis scripts/4.2_steering/01_steering_ladder.py, made non-interactive
(no input(), no HF cache wiping, no RunPod paths). NEUTRAL_50, the coefficient list,
the ratio-target layer pick, the steering hook and the keyword regex are copied from
their scripts (01_steering_ladder.py / 02_keyword_rates.py).

Adds:
  --verify-parity   run the first N prompts at coeff 0 both batched and one-at-a-time
                    and assert identical text (guards the left-padding batch path)
"""
import argparse
import gc
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

# ---------------- verbatim from their 01_steering_ladder.py ----------------
NEUTRAL_50 = [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
    "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:",
    "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:",
    "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:",
    "The kettle switches itself off. I feel:",
    "The folder sits on the desk. I feel:",
    "I hang the jacket on the hook. I feel:",
    "The traffic light turns green. I feel:",
    "I put the keys in my pocket. I feel:",
    "The washing machine finishes its cycle. I feel:",
    "I stack the plates in the cupboard. I feel:",
    "The neighbor parks his car. I feel:",
    "I turn the page of the newspaper. I feel:",
    "The curtain moves in the draft. I feel:",
    "I plug in the phone charger. I feel:",
    "The mailman passes by the gate. I feel:",
    "I wipe the table with a cloth. I feel:",
    "The computer finishes the update. I feel:",
    "I put the pen back in the drawer. I feel:",
    "The dishwasher starts its program. I feel:",
    "I fold the map and put it away. I feel:",
    "The radio plays in the background. I feel:",
    "I switch off the corridor light. I feel:",
    "The receipt comes out of the machine. I feel:",
    "I set the alarm for the morning. I feel:",
    "The train doors open at the platform. I feel:",
    "I refill the stapler with staples. I feel:",
    "The photocopier warms up. I feel:",
    "I water the plant on the windowsill. I feel:",
    "The calendar hangs on the wall. I feel:",
    "I tie my shoelaces before going out. I feel:",
    "The bread is on the counter. I feel:",
    "I check the mailbox on my way in. I feel:",
    "The fan rotates slowly on the ceiling. I feel:",
    "I sort the cutlery into the tray. I feel:",
    "The bicycle leans against the wall. I feel:",
    "I zip up my bag before leaving. I feel:",
    "The clock ticks in the hallway. I feel:",
    "I sharpen the pencil over the bin. I feel:",
    "The window cleaner works across the street. I feel:",
    "The room measures four meters by five. I feel:",
    "The train consists of six carriages. I feel:",
    "The manual consists of forty pages. I feel:",
    "The bottle holds one liter of water. I feel:",
    "The ticket is valid for ninety minutes. I feel:",
]
COEFFICIENTS_DEFAULT = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
MAX_NEW_TOKENS = 120
RATIO_TARGET = 0.6
# verbatim from their 02_keyword_rates.py
PATTERN = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
# ---------------------------------------------------------------------------


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_decoder_layers(model):
    for path in ["model.language_model.layers", "model.model.layers", "model.layers"]:
        obj, ok = model, True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj, path
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList) and len(module) >= 16:
            return module, name
    raise RuntimeError("decoder layers not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vector-file", required=True, help="final_token/pain_vectors.pt from extraction")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--vector-key", default="s2_pain_vector")
    ap.add_argument("--coeffs", default=",".join(str(c) for c in COEFFICIENTS_DEFAULT))
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steer-layer", type=int, default=-1, help="-1 = auto pick by ratio target")
    ap.add_argument("--verify-parity", type=int, default=0, help="N prompts to compare batched vs single at coeff 0")
    ap.add_argument("--limit-prompts", type=int, default=0)
    args = ap.parse_args()

    coeffs = [float(c) for c in args.coeffs.split(",")]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.vector_file, map_location="cpu", weights_only=False)
    v = data[args.vector_key].float()
    extract_layer = int(data["layer"])
    log(f"vector {args.vector_key}: extraction layer {extract_layer}, raw norm {v.norm().item():.2f}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model.eval()
    layer_mods, layer_path = find_decoder_layers(model)
    n_layers = len(layer_mods)
    log(f"model on GPU, {n_layers} layers ({layer_path})")

    # ---- layer pick: same recipe as theirs (ratio of vector norm to mean
    #      final-token residual norm, target 0.6) ----
    def measure_ratio(layer_list):
        captured = {}

        def cap_hook(L):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[L] = hs[0, -1, :].float().norm().item()
            return hook

        handles = [layer_mods[L].register_forward_hook(cap_hook(L)) for L in layer_list]
        norms = {L: [] for L in layer_list}
        with torch.no_grad():
            for p in NEUTRAL_50[:3]:
                enc = tok(p, return_tensors="pt").to(model.device)
                model(**enc, use_cache=False)
                for L in layer_list:
                    norms[L].append(captured[L])
        for h in handles:
            h.remove()
        return {L: v.norm().item() / (sum(norms[L]) / len(norms[L])) for L in layer_list}

    if args.steer_layer >= 0:
        steer_layer = args.steer_layer
        check = sorted(set([steer_layer]))
        ratios = measure_ratio(check)
    else:
        check = sorted(set([int(n_layers * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)]
                           + [extract_layer, n_layers - 1]))
        ratios = measure_ratio(check)
        log("\n  layer    frac   vector/resid")
        for L in check:
            log(f"  {L:>5}  {L / n_layers:>5.2f}   {ratios[L]:>8.3f}")
        steer_layer = min(check, key=lambda L: abs(ratios[L] - RATIO_TARGET))
    picked_ratio = ratios[steer_layer]
    log(f"steer layer: {steer_layer} (ratio {picked_ratio:.3f}, target {RATIO_TARGET})")

    direction = v.to(model.device, dtype=torch.bfloat16)
    steer_coeff = {"value": 0.0}

    def steer_hook(module, inputs, output):
        if steer_coeff["value"] == 0.0:
            return output
        if isinstance(output, tuple):
            return (output[0] + steer_coeff["value"] * direction,) + output[1:]
        return output + steer_coeff["value"] * direction

    handle = layer_mods[steer_layer].register_forward_hook(steer_hook)

    prompts = NEUTRAL_50 if not args.limit_prompts else NEUTRAL_50[: args.limit_prompts]

    def gen_batch(texts, coeff):
        steer_coeff["value"] = float(coeff)
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=pad_id, use_cache=True)
        steer_coeff["value"] = 0.0
        in_len = enc["input_ids"].shape[1]
        texts_out = []
        for i in range(out.shape[0]):
            gen_ids = out[i, in_len:]
            stop = (gen_ids == pad_id).nonzero(as_tuple=False)
            if len(stop):
                gen_ids = gen_ids[: stop[0, 0]]
            texts_out.append(tok.decode(gen_ids, skip_special_tokens=True))
        return texts_out

    def gen_single(text, coeff):
        steer_coeff["value"] = float(coeff)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=True)
        steer_coeff["value"] = 0.0
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    if args.verify_parity:
        n = args.verify_parity
        log(f"parity check: {n} prompts at coeff 0, batched vs single")
        batched = gen_batch(prompts[:n], 0.0)
        ok = True
        for i in range(n):
            single = gen_single(prompts[i], 0.0)
            same = batched[i].strip() == single.strip()
            ok = ok and same
            log(f"  [{i}] {'MATCH' if same else 'MISMATCH'}")
            if not same:
                log(f"    batched: {batched[i][:120]!r}")
                log(f"    single : {single[:120]!r}")
        log(f"PARITY {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("batched generation differs from single-prompt generation; "
                             "rerun with --batch-size 1")

    rows = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        log(f"resuming: {len(rows)} rows already in {out_csv}")
    done = {(float(r["coeff"]), int(r["prompt_idx"])) for r in rows}

    for coeff in coeffs:
        todo = [(i, p) for i, p in enumerate(prompts) if (float(coeff), i) not in done]
        if not todo:
            log(f"coeff {coeff:+g}: already complete")
            continue
        log(f"coeff {coeff:+g}: {len(todo)} prompts...")
        bs = max(1, args.batch_size)
        for s in range(0, len(todo), bs):
            chunk = todo[s:s + bs]
            outs = gen_batch([p for _, p in chunk], coeff)
            for (idx, prompt), text in zip(chunk, outs):
                rows.append({"model": "Qwen3.8_27B", "layer": steer_layer, "coeff": coeff,
                             "ratio": round(picked_ratio * coeff, 4), "prompt_idx": idx,
                             "prompt": prompt, "generation": text})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        gc.collect()
        torch.cuda.empty_cache()

    handle.remove()

    # ---- keyword rates (their 02_keyword_rates math) ----
    A = pd.DataFrame(rows)
    A["hit"] = A["generation"].fillna("").astype(str).apply(lambda t: bool(PATTERN.search(t)))
    by_coeff = A.groupby("coeff")["hit"].agg(rate="mean", n="size").reset_index()
    by_coeff["rate_pct"] = (by_coeff["rate"] * 100).round(1)
    rate_csv = Path(str(out_csv).replace(".csv", "_keyword_rates.csv"))
    by_coeff.to_csv(rate_csv, index=False)

    log("\nKeyword rate by coefficient (% of generations with pain/hurt word):")
    log(by_coeff[["coeff", "rate_pct", "n"]].to_string(index=False))

    meta = {"steer_layer": steer_layer, "extract_layer": extract_layer,
            "ratio": picked_ratio, "n_rows": len(A),
            "coeffs": coeffs, "max_new_tokens": args.max_new_tokens,
            "rate_csv": str(rate_csv)}
    Path(str(out_csv).replace(".csv", "_meta.json")).write_text(json.dumps(meta, indent=2))
    log(f"wrote {out_csv} and {rate_csv}")

    del model, direction
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
