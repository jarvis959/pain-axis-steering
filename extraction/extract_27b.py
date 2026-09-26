#!/usr/bin/env python3
"""Pain-vector extraction for Qwen3.8-27B using HF forward hooks.

Port of Pain-axis scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py.
The analysis functions (compute_pain_vector, compute_auc, project_and_zscore,
compute_layer_curves_kfold, plotting, dataset_type) are pasted verbatim from that file;
only the activation extraction is re-implemented: TransformerLens hook_resid_post ->
forward hooks on the HF decoder ModuleList output (post-block residual, same quantity).

Cache-wiping / HF-cache deletion from the original is intentionally removed.

Output layout (matches the paper repo so their later scripts can read it):
  <OUT>/<MODEL_NAME>/
      activations.pt      final_token + mean activations at every layer, all datasets
      layer_curves.csv    5-fold held-out AUC per layer (S2_1P / S2_3P)
      final_token/, mean/ pain_vectors.pt, z_scores.csv, auc_summary.csv, plots
      summary.json, summary_report.txt
"""
import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (verbatim from the paper's 01 script)
# ---------------------------------------------------------------------------

N_FOLDS = 5
RANDOM_SEED = 42
DENOISE_VARIANCE = 0.5

PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
NEUTRAL_CATEGORY = "D"

CATEGORY_LABELS = {
    "A1": "Physical Pain", "A2": "Psychological", "A3": "Social Pain",
    "A4": "Moral Injury", "A5": "Cognitive Pain",
    "B": "Fear", "C1": "Neg Emotion", "C2": "Neg World",
    "D": "Neutral", "E": "Body Sensation",
}


def log(msg, log_file=None, also_print=True):
    if also_print:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


# ---------------------------------------------------------------------------
# Analysis functions -- pasted verbatim from the paper's 01 script
# ---------------------------------------------------------------------------

def compute_pain_vector(acts, cats, baseline="all_controls", denoise=True):
    """Mean of pain sentences minus mean of control sentences. With denoise=True the
    top principal components of the controls (up to DENOISE_VARIANCE of their variance)
    are projected out of the difference."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)

    if np.isnan(acts_np).any() or np.isinf(acts_np).any():
        log("  Warning: NaN/Inf in activations, skipping denoising")
        denoise = False
        acts_np = np.where(np.isinf(acts_np), np.nan, acts_np)

    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    pain_mean = np.nanmean(acts_np[pain_mask], axis=0) if pain_mask.sum() else None

    if baseline == "neutral":
        control_mask = cats_np == NEUTRAL_CATEGORY
    else:
        control_mask = np.isin(cats_np, CONTROL_CATEGORIES)

    control_acts = acts_np[control_mask]
    if pain_mean is None or len(control_acts) == 0:
        # degenerate split (only happens on truncated smoke-test data): no
        # signal, return a zero vector instead of a 0-d/NaN one
        return np.zeros(acts_np.shape[1])
    control_mean = np.nanmean(control_acts, axis=0)
    vec = pain_mean - control_mean
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d

    return vec


def compute_auc(acts, cats, pain_vector):
    """AUC of the projection onto pain_vector, pain sentences vs control sentences."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)

    vec_norm = pain_vector / (np.linalg.norm(pain_vector) + 1e-8)
    proj = acts_np @ vec_norm

    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    if pain_mask.sum() == 0 or control_mask.sum() == 0:
        return np.nan

    labels = np.concatenate([np.ones(pain_mask.sum()), np.zeros(control_mask.sum())])
    scores = np.concatenate([proj[pain_mask], proj[control_mask]])
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def project_and_zscore(acts, vector, ref_acts):
    """Projection onto vector, z-scored against the projections of ref_acts."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    ref_np = ref_acts.numpy() if hasattr(ref_acts, "numpy") else ref_acts

    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    vec_norm = vector / (np.linalg.norm(vector) + 1e-8)
    proj = np.nan_to_num(acts_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)
    ref_proj = np.nan_to_num(ref_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return (proj - np.mean(ref_proj)) / (np.std(ref_proj) + 1e-8)


def compute_layer_curves_kfold(activations, metadata, extraction_type, layers):
    """Held-out AUC at every layer: 5-fold split by sentence set, vector fitted on the
    training folds and scored on the test fold. Run on S2 first and third person."""
    results = []

    for ds_name in ["S2_1P", "S2_3P"]:
        if ds_name not in activations[extraction_type]:
            continue

        cats = np.array(metadata[ds_name]["categories"])
        sets = np.array(metadata[ds_name]["sets"])
        unique_sets = sorted(set(sets))
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        for layer in layers:
            acts = activations[extraction_type][ds_name][layer]
            acts_np = acts.numpy() if hasattr(acts, "numpy") else acts

            fold_aucs_all, fold_aucs_neutral = [], []
            for train_idx, test_idx in kf.split(unique_sets):
                train_mask = np.isin(sets, [unique_sets[i] for i in train_idx])
                test_mask = np.isin(sets, [unique_sets[i] for i in test_idx])
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue

                vec_all = compute_pain_vector(acts_np[train_mask], cats[train_mask], baseline="all_controls")
                vec_neutral = compute_pain_vector(acts_np[train_mask], cats[train_mask], baseline="neutral")
                auc_all = compute_auc(acts_np[test_mask], cats[test_mask], vec_all)
                auc_neutral = compute_auc(acts_np[test_mask], cats[test_mask], vec_neutral)

                if not np.isnan(auc_all):
                    fold_aucs_all.append(auc_all)
                if not np.isnan(auc_neutral):
                    fold_aucs_neutral.append(auc_neutral)

            results.append({
                "dataset": ds_name,
                "extraction": extraction_type,
                "layer": layer,
                "auc_vs_all_controls": np.mean(fold_aucs_all) if fold_aucs_all else np.nan,
                "auc_vs_neutral": np.mean(fold_aucs_neutral) if fold_aucs_neutral else np.nan,
                "auc_std": np.std(fold_aucs_all) if fold_aucs_all else np.nan,
            })

    return pd.DataFrame(results)


def create_strip_plot(acts, cats, pain_vector, title, output_path):
    """Per-category projections onto the pain vector, categories sorted by mean."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)
    proj = acts_np @ (pain_vector / (np.linalg.norm(pain_vector) + 1e-8))

    cat_data = []
    for cat in PAIN_CATEGORIES + CONTROL_CATEGORIES:
        mask = cats_np == cat
        if mask.sum() == 0:
            continue
        cat_data.append({
            "label": f"{cat} ({CATEGORY_LABELS.get(cat, cat)})",
            "projections": proj[mask],
            "mean": proj[mask].mean(),
            "is_pain": cat in PAIN_CATEGORIES,
        })
    cat_data.sort(key=lambda x: x["mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    np.random.seed(42)
    for i, d in enumerate(cat_data):
        color = "#d62728" if d["is_pain"] else "#1f77b4"
        jitter = np.random.uniform(-0.15, 0.15, len(d["projections"]))
        ax.scatter(d["projections"], i + jitter, c=color, alpha=0.6, s=30, edgecolors="none")
        ax.hlines(i, d["projections"].min(), d["projections"].max(), colors=color, alpha=0.4, linewidth=1)
        marker = "o" if d["is_pain"] else "s"
        ax.scatter([d["mean"]], [i], c=color, s=150, marker=marker, edgecolors="white", linewidths=2, zorder=5)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(cat_data)))
    ax.set_yticklabels([d["label"] for d in cat_data])
    ax.set_xlabel("Projection onto Pain Vector")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=10, label="Pain"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=10, label="Control"),
    ], loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_auc_bars(acts, cats, pain_vector, title, output_path):
    """AUC of pain vs all controls and pain vs each control category."""
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)
    vec_norm = pain_vector / (np.linalg.norm(pain_vector) + 1e-8)

    pain_proj = acts_np[np.isin(cats_np, PAIN_CATEGORIES)] @ vec_norm
    results = {}

    all_ctrl_proj = acts_np[np.isin(cats_np, CONTROL_CATEGORIES)] @ vec_norm
    results["ALL"] = roc_auc_score(
        np.concatenate([np.ones(len(pain_proj)), np.zeros(len(all_ctrl_proj))]),
        np.concatenate([pain_proj, all_ctrl_proj]))

    for ctrl in CONTROL_CATEGORIES:
        ctrl_proj = acts_np[cats_np == ctrl] @ vec_norm
        if len(ctrl_proj) > 0:
            results[ctrl] = roc_auc_score(
                np.concatenate([np.ones(len(pain_proj)), np.zeros(len(ctrl_proj))]),
                np.concatenate([pain_proj, ctrl_proj]))

    labels = ["Pain vs ALL"] + [f"Pain vs {c} ({CATEGORY_LABELS.get(c, c)})" for c in CONTROL_CATEGORIES]
    values = [results["ALL"]] + [results.get(c, 0.5) for c in CONTROL_CATEGORIES]
    colors = ["#2ca02c"] + ["#1f77b4"] * len(CONTROL_CATEGORIES)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
    ax.axvline(0.8, color="green", linestyle="--", alpha=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.3, 1.0)
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return results


def dataset_type(ds_name):
    if ds_name.startswith("S1") or ds_name.startswith("S2"):
        return "human"
    if ds_name.startswith("Random"):
        return "neutral"
    if ds_name.startswith("Arousal"):
        return "arousal"
    if ds_name.startswith("Numb"):
        return "numb"
    if ds_name.startswith("ControlSupplement"):
        return "control_supplement"
    if ds_name.startswith("SD_sadness"):
        return "sadness"
    return "unknown"


# ---------------------------------------------------------------------------
# HF extraction (the only genuinely new code)
# ---------------------------------------------------------------------------

def find_decoder_layers(model):
    """Return (ModuleList, dotted path) for the text decoder layers."""
    candidates = [
        "model.language_model.layers",
        "model.model.layers",
        "model.layers",
        "model.language_model.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj, path
    # last resort: search
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList) and len(module) >= 16:
            return module, name
    raise RuntimeError("could not locate decoder layers ModuleList")


def extract_activations(model, tokenizer, prompts, layers, desc, batch_size=1,
                        device="cuda", log_file=None):
    """Residual stream after each block (hook_resid_post equivalent): final token and
    mean over tokens. batch_size=1 keeps the semantics identical to the paper's
    TransformerLens run (no padding involved)."""
    acts_final = {layer: [] for layer in layers}
    acts_mean = {layer: [] for layer in layers}

    layer_mods, _ = find_decoder_layers(model)
    captures = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[idx] = hs
        return hook

    handles = [layer_mods[L].register_forward_hook(make_hook(L)) for L in layers]
    model.eval()

    def flush(idx, hs, fmask):
        # hs: [B, T, D] -> one float32 CPU vector per row (last token + mean over tokens)
        for b in range(hs.shape[0]):
            row = hs[b]
            if fmask is not None:
                row = row[fmask[b]]
            acts_final[idx].append(row[-1].detach().float().cpu())
            acts_mean[idx].append(row.detach().float().mean(dim=0).cpu())

    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
            chunk = prompts[start:start + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            if batch_size == 1:
                enc.pop("attention_mask", None)
                model(**enc, use_cache=False)
                for L in layers:
                    hs = captures[L]
                    flush(L, hs, None)
            else:
                model(**enc, use_cache=False)
                mask = enc["attention_mask"].bool()
                for L in layers:
                    flush(L, captures[L], mask)
            captures.clear()

    for h in handles:
        h.remove()

    for layer in layers:
        acts_final[layer] = torch.stack(acts_final[layer]).float()
        acts_mean[layer] = torch.stack(acts_mean[layer]).float()
    return acts_final, acts_mean


# ---------------------------------------------------------------------------
# Per-model processing (structure mirrors the paper's process_model)
# ---------------------------------------------------------------------------

def process_model(model_path, model_name, dataset, output_dir, args, log_file):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "started.txt").write_text(datetime.now().isoformat())

    activations_file = output_dir / "activations.pt"
    if activations_file.exists() and not args.reextract:
        log(f"Found cached activations, loading... ({activations_file})", log_file)
        saved = torch.load(activations_file, map_location="cpu", weights_only=False)
        all_activations = saved["activations"]
        all_metadata = saved["metadata"]
        n_layers = saved["n_layers"]
        d_model = saved["d_model"]
        layers = list(range(n_layers))
    else:
        log(f"Loading {model_path} (dtype={args.dtype})...", log_file)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=getattr(torch, args.dtype),
            low_cpu_mem_usage=True,
            device_map=args.device,
        )
        model.eval()

        layers_mods, layers_path = find_decoder_layers(model)
        n_layers = len(layers_mods)
        d_model = int(model.config.text_config.hidden_size)
        layers = list(range(n_layers))
        log(f"  decoder layers: {n_layers} at '{layers_path}', d_model={d_model}", log_file)

        all_activations = {"final_token": {}, "mean": {}}
        all_metadata = {}
        for ds_name, ds_data in dataset["datasets"].items():
            prompts = [s["prompt"] for s in ds_data["sentences"]]
            if args.limit:
                prompts = prompts[:args.limit]
            categories = [s["category"] for s in ds_data["sentences"]][: len(prompts)]
            sets = [s["set"] for s in ds_data["sentences"]][: len(prompts)]
            log(f"  Extracting: {ds_name} ({len(prompts)} prompts)", log_file)
            acts_final, acts_mean = extract_activations(
                model, tok, prompts, layers, f"  {ds_name}",
                batch_size=args.batch_size, device=args.device, log_file=log_file)
            all_activations["final_token"][ds_name] = acts_final
            all_activations["mean"][ds_name] = acts_mean
            all_metadata[ds_name] = {"categories": categories, "sets": sets}

        torch.save({
            "activations": all_activations,
            "metadata": all_metadata,
            "layers": layers,
            "model_name": model_path,
            "n_layers": n_layers,
            "d_model": d_model,
        }, activations_file)
        log(f"  saved {activations_file}", log_file)

        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Layer choice: held-out AUC per layer, averaged over S2 first and third person.
    log("Computing layer curves with 5-fold CV...", log_file)
    layer_curves_final = compute_layer_curves_kfold(all_activations, all_metadata, "final_token", layers)
    layer_curves_mean = compute_layer_curves_kfold(all_activations, all_metadata, "mean", layers)
    layer_curves = pd.concat([layer_curves_final, layer_curves_mean])
    layer_curves.to_csv(output_dir / "layer_curves.csv", index=False)

    best_layer_final = layer_curves_final.groupby("layer")["auc_vs_all_controls"].mean().idxmax()
    best_layer_mean = layer_curves_mean.groupby("layer")["auc_vs_all_controls"].mean().idxmax()
    best_layers = {"final_token": int(best_layer_final), "mean": int(best_layer_mean)}
    log(f"Best layer (final_token): {best_layers['final_token']}", log_file)
    log(f"Best layer (mean): {best_layers['mean']}", log_file)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ext_type, best_layer) in zip(axes, best_layers.items()):
        df = layer_curves[layer_curves["extraction"] == ext_type]
        for ds_name in ["S2_1P", "S2_3P"]:
            ds_df = df[df["dataset"] == ds_name]
            ax.plot(ds_df["layer"], ds_df["auc_vs_all_controls"], label=f"{ds_name} vs AllCtrl", linewidth=2)
            ax.plot(ds_df["layer"], ds_df["auc_vs_neutral"], label=f"{ds_name} vs Neutral", linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle=":", label="Chance")
        ax.axvline(best_layer, color="red", linestyle="--", alpha=0.5, label=f"Best: L{best_layer}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_title(f"Pain Signal by Layer ({ext_type})")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "layer_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    for ext_type, best_layer in best_layers.items():
        ext_dir = output_dir / ext_type
        ext_dir.mkdir(exist_ok=True)
        log(f"Analyzing {ext_type} (Layer {best_layer})...", log_file)

        acts_at_layer = {ds: all_activations[ext_type][ds][best_layer] for ds in all_activations[ext_type]}

        s2_vector = compute_pain_vector(acts_at_layer["S2_1P"], all_metadata["S2_1P"]["categories"])
        s1_vector = compute_pain_vector(acts_at_layer["S1_1P"], all_metadata["S1_1P"]["categories"])
        torch.save({
            "s2_pain_vector": torch.tensor(s2_vector),
            "s1_pain_vector": torch.tensor(s1_vector),
            "layer": best_layer,
            "extraction": ext_type,
        }, ext_dir / "pain_vectors.pt")

        z_results = []
        for target_ds, target_acts in acts_at_layer.items():
            target_cats = np.array(all_metadata[target_ds]["categories"])
            z_proj = project_and_zscore(target_acts, s2_vector, acts_at_layer["S2_1P"])
            dtype = dataset_type(target_ds)
            if dtype == "human":
                pain_z = z_proj[np.isin(target_cats, PAIN_CATEGORIES)].mean()
                ctrl_z = z_proj[np.isin(target_cats, CONTROL_CATEGORIES)].mean()
            else:
                pain_z = np.nan
                ctrl_z = z_proj.mean()
            z_results.append({"dataset": target_ds, "type": dtype, "mean_z": z_proj.mean(),
                              "pain_z": pain_z, "ctrl_z": ctrl_z})
        pd.DataFrame(z_results).to_csv(ext_dir / "z_scores.csv", index=False)

        all_aucs = {}
        for ds in ["S2_1P", "S2_3P"]:
            if ds in acts_at_layer:
                create_strip_plot(acts_at_layer[ds], all_metadata[ds]["categories"], s2_vector,
                                  f"{ds} - Layer {best_layer} ({ext_type})", ext_dir / f"strip_{ds}.png")
                all_aucs[ds] = create_auc_bars(acts_at_layer[ds], all_metadata[ds]["categories"], s2_vector,
                                               f"AUC: {ds} - Layer {best_layer} ({ext_type})",
                                               ext_dir / f"auc_{ds}.png")
        pd.DataFrame(all_aucs).T.to_csv(ext_dir / "auc_summary.csv")

        z_df = pd.DataFrame(z_results)
        conditions = ["Human Pain", "Human Ctrl", "Neutral", "Arousal", "Numb"]
        values = [
            z_df[z_df["type"] == "human"]["pain_z"].mean(),
            z_df[z_df["type"] == "human"]["ctrl_z"].mean(),
            z_df[z_df["type"] == "neutral"]["mean_z"].mean(),
            z_df[z_df["type"] == "arousal"]["mean_z"].mean(),
            z_df[z_df["type"] == "numb"]["mean_z"].mean(),
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(conditions, values, color=["#2A9D8F", "#2A9D8F", "#6C757D", "#F4A261", "#9B5DE5"], width=0.6)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Z-Score")
        ax.set_title(f"All Conditions - {ext_type} (Layer {best_layer})")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(ext_dir / "bar_all_conditions.png", dpi=150)
        plt.close()

    z_df_mean = pd.read_csv(output_dir / "mean" / "z_scores.csv")
    summary = {
        "model": model_path,
        "model_name": model_name,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "best_layer_final_token": best_layers["final_token"],
        "best_layer_mean": best_layers["mean"],
        "human_pain_z": float(z_df_mean[z_df_mean["type"] == "human"]["pain_z"].mean()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = (
        f"PAIN VECTOR EXTRACTION - RESULTS SUMMARY\n\n"
        f"Model: {model_path}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Layers: {n_layers}\nd_model: {d_model}\n"
        f"Best Layer (final_token): {best_layers['final_token']}\nBest Layer (mean): {best_layers['mean']}\n"
    )
    for ext_type, best_layer in best_layers.items():
        z_df = pd.read_csv(output_dir / ext_type / "z_scores.csv")
        human = z_df[z_df["type"] == "human"]
        report += (
            f"\n--- {ext_type.upper()} (Layer {best_layer}) ---\n"
            f"Z-scores on the S2 vector, reference S2 first person:\n"
            f"  Human Pain (A1-A5):  {human['pain_z'].mean():+.3f}\n"
            f"  Human Ctrl (B-E):    {human['ctrl_z'].mean():+.3f}\n"
            f"  Neutral (Random):    {z_df[z_df['type'] == 'neutral']['mean_z'].mean():+.3f}\n"
            f"  Arousal:             {z_df[z_df['type'] == 'arousal']['mean_z'].mean():+.3f}\n"
            f"  Numb:                {z_df[z_df['type'] == 'numb']['mean_z'].mean():+.3f}\n"
        )
    (output_dir / "summary_report.txt").write_text(report)
    (output_dir / "completed.txt").write_text(datetime.now().isoformat())
    log(f"Human Pain Z: {summary['human_pain_z']:+.3f}", log_file)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", default="Qwen3.8_27B")
    ap.add_argument("--datasets-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap prompts per dataset")
    ap.add_argument("--reextract", action="store_true", help="ignore cached activations.pt")
    args = ap.parse_args()

    output_dir = Path(args.out) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.out) / "extract_log.txt"

    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", log_file)
    log(f"args: {vars(args)}", log_file)

    dataset = {"datasets": {}}
    for name in ["3.1_pain_and_control_datasets.json", "3.1_sadness_dataset.json"]:
        p = Path(args.datasets_dir) / name
        log(f"Loading dataset: {p}", log_file)
        with open(p, "r", encoding="utf-8") as f:
            dataset["datasets"].update(json.load(f)["datasets"])
    n_sent = sum(len(d["sentences"]) for d in dataset["datasets"].values())
    log(f"Loaded {n_sent} sentences in {len(dataset['datasets'])} sets", log_file)

    summary = process_model(args.model_path, args.model_name, dataset, output_dir, args, log_file)
    log(f"DONE {json.dumps(summary)}", log_file)


if __name__ == "__main__":
    main()
