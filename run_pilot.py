"""
run_pilot.py — Representation-Mediated Reasoning (RMR) pilot experiment.

Hypothesis (RMR)
────────────────
  "Reasoning capability in LLMs arises from structured internal
   representations, not from properties unique to natural language.
   Natural language is a privileged but non-essential representation."

Operational prediction
──────────────────────
  In a regression predicting pairwise representational distance:

    y[i,j] ~ β₀ + β₁·same_concept + β₂·same_language_type
               + β₃·|token_count_diff| + β₄·|entropy_diff| + β₅·|ttr_diff|

  RMR predicts: |β₁| dominates; |β₂| is NOT dominant after controlling
  for β₃–β₅ (structural bias features).

Pipeline
────────
  1. Build stimuli (5 concepts × 4 representations = 20 texts)
  2. Extract per-layer hidden states (with cache)
  3. Compute structural bias features for each stimulus
  4. Run analyses:
       M1a  Extended RSA   (4 theoretical RDMs)
       M1b  Original RSA   (2 theoretical RDMs, for comparison)
       M2   Cross-form linear probe transfer
       M3   Silhouette curves
       CKA  Cross-form alignment
       REG  Bias regression (core RMR test)
  5. Print summary + RMR hypothesis verdict
  6. Save ALL numerical results to results/pilot_numbers.json
  7. Produce 10 publication-ready figures

Usage
─────
  python run_pilot.py                   # gpt2-xl (default in config.py)
  python run_pilot.py --model gpt2      # quick pilot
  python run_pilot.py --no-cache        # force re-extraction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.stimuli import build_stimuli, get_labels, CONCEPT_NAMES, FORM_NAMES
from src.extractor import RepresentationExtractor
from src.metrics import (
    rsa_all_layers,
    rsa_all_layers_v2,
    cross_probe_all_layers,
    cross_probe_summary,
    silhouette_all_layers,
    find_crossover_layer,
    cka_all_layers,
    cka_summary,
    bias_regression_all_layers,
)
from src.rep_bias import (
    compute_bias_features,
    build_bias_rdm,
    build_language_type_rdm,
    get_language_type_labels,
    get_normalized_bias_matrix,
    bias_summary_by_form,
)
from src.visualize import produce_all_figures_v2


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RMR pilot experiment")
    p.add_argument("--model",       default=None)
    p.add_argument("--no-cache",    action="store_true")
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def step_stimuli():
    print("\n" + "═" * 62)
    print("STEP 1 — Build stimuli")
    print("═" * 62)
    stimuli = build_stimuli()
    c_labels, f_labels = get_labels(stimuli)
    print(f"  {len(stimuli)} stimuli  |  "
          f"{len(set(c_labels))} concepts × {len(set(f_labels))} forms")
    for i, s in enumerate(stimuli):
        print(f"  [{i:02d}] {s.concept_name:28s} | {s.form_name:10s} | "
              f"{s.text[:52].strip()!r}…")
    return stimuli, c_labels, f_labels


def step_extract(stimuli, model_name, cache_path, force=False):
    print("\n" + "═" * 62)
    print("STEP 2 — Extract hidden-state representations")
    print("═" * 62)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    npz = cache_path + ".npz"
    if not force and os.path.exists(npz):
        data = np.load(npz, allow_pickle=True)
        if str(data["model_name"]) == model_name:
            act = data["activations"]
            print(f"  Loaded from cache: {npz}  shape={act.shape}")
            return act

    extractor = RepresentationExtractor(
        model_name, device=config.DEVICE, dtype=config.DTYPE
    )
    extractor.load()
    t0 = time.time()
    act = extractor.extract_all(stimuli)
    elapsed = time.time() - t0
    nl, hs = extractor.n_layers, extractor.hidden_size
    extractor.unload()
    np.savez_compressed(
        cache_path, activations=act,
        model_name=np.array(model_name),
        n_stimuli=np.array(len(stimuli)),
        n_layers=np.array(nl), hidden_size=np.array(hs),
    )
    print(f"  Done in {elapsed:.1f}s → {npz}  shape={act.shape}")
    return act


def step_bias(stimuli, model_name):
    print("\n" + "═" * 62)
    print("STEP 3 — Compute representation bias features")
    print("═" * 62)
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Tokenizer loaded: {model_name}")
    except Exception as e:
        print(f"  Tokenizer unavailable ({e}); falling back to char-level stats")

    bias_df = compute_bias_features(stimuli, tokenizer)
    bias_rdm = build_bias_rdm(bias_df)
    lang_rdm = build_language_type_rdm(stimuli)
    lang_labels = get_language_type_labels(stimuli)
    bias_matrix = get_normalized_bias_matrix(bias_df)
    bias_summary = bias_summary_by_form(bias_df)

    print("\n  Bias features by form:")
    print(bias_summary.round(3).to_string())
    print()
    return bias_df, bias_rdm, lang_rdm, lang_labels, bias_matrix, bias_summary


def step_analyse(activations, c_labels, f_labels,
                 bias_rdm, lang_rdm, lang_labels, bias_matrix):
    print("\n" + "═" * 62)
    print("STEP 4 — Run analyses")
    print("═" * 62)
    N, L, D = activations.shape
    print(f"  activations: ({N}, {L}, {D})")

    print("  [M1a] Extended RSA (4 RDMs) …", end=" ", flush=True)
    rsa_v2 = rsa_all_layers_v2(activations, c_labels, f_labels,
                                bias_rdm, lang_rdm)
    print("done")

    print("  [M1b] Original RSA (2 RDMs) …", end=" ", flush=True)
    rsa_orig = rsa_all_layers(activations, c_labels, f_labels)
    print("done")

    print("  [M2]  Cross-form linear probe …", end=" ", flush=True)
    probe_t = cross_probe_all_layers(activations, c_labels, f_labels)
    probe_s = cross_probe_summary(probe_t)
    print("done")

    print("  [M3]  Silhouette …", end=" ", flush=True)
    sil = silhouette_all_layers(activations, c_labels, f_labels)
    print("done")

    print("  [CKA] Cross-form CKA …", end=" ", flush=True)
    cka_t = cka_all_layers(activations, f_labels)
    cka_m = cka_summary(cka_t)
    print("done")

    bias_feat_names = ["token_count_diff", "entropy_diff", "ttr_diff"]
    print("  [REG] Bias regression …", end=" ", flush=True)
    reg = bias_regression_all_layers(
        activations, c_labels, lang_labels, bias_matrix, bias_feat_names
    )
    print("done")

    crossover = find_crossover_layer(sil["sil_concept"], sil["sil_form"])
    return rsa_v2, rsa_orig, probe_t, probe_s, sil, cka_m, reg, crossover


def step_summary(activations, rsa_v2, rsa_orig, probe_s,
                 sil, cka_m, reg, crossover, model_name):
    N, L, D = activations.shape
    last = L - 1

    print("\n" + "═" * 62)
    print("STEP 5 — Results summary (RMR hypothesis test)")
    print("═" * 62)

    sep = "─" * 62
    print(f"\n  Model : {model_name}   Layers: {L}   Hidden: {D}")
    print(f"\n  {sep}")
    print(f"  {'Metric':<40} {'Best L':>6}  {'Value':>8}")
    print(f"  {sep}")

    def row(name, layer, val):
        ls = str(layer) if layer >= 0 else "none"
        vs = f"{val:.4f}" if not np.isnan(val) else "—"
        print(f"  {name:<40} {ls:>6}  {vs:>8}")

    # RSA
    row("M1a RSA ρ(concept) max",
        int(np.argmax(rsa_v2["corr_concept"])),
        float(np.max(rsa_v2["corr_concept"])))
    row("M1a RSA ρ(language_type) max",
        int(np.argmax(rsa_v2["corr_language_type"])),
        float(np.max(rsa_v2["corr_language_type"])))
    row("M1a RSA ρ(bias) max",
        int(np.argmax(rsa_v2["corr_bias"])),
        float(np.max(rsa_v2["corr_bias"])))

    # Probe
    best_probe = int(np.argmax(probe_s["off_diag_mean"]))
    row("M2  Cross-form probe max",
        best_probe, float(np.max(probe_s["off_diag_mean"])))
    row("M2  Within-form probe max",
        int(np.argmax(probe_s["diagonal_mean"])),
        float(np.max(probe_s["diagonal_mean"])))

    # Sil
    row("M3  Silhouette(concept) max",
        int(np.argmax(sil["sil_concept"])),
        float(np.max(sil["sil_concept"])))
    row("M3  Silhouette crossover layer", crossover, float("nan"))

    # CKA
    row("CKA cross-form mean max",
        int(np.argmax(cka_m)), float(np.max(cka_m)))

    # Regression
    bc = reg["same_concept"][last]
    bl = reg["same_language_type"][last]
    r2 = reg["r2"][last]
    row("REG β(same_concept) @ final layer",    last, float(bc))
    row("REG β(same_language_type) @ final L",  last, float(bl))
    row("REG R² @ final layer",                 last, float(r2))
    print(f"  {sep}")

    # ── RMR verdict ─────────────────────────────────────────────────────
    print("\n  ── RMR HYPOTHESIS TEST ─────────────────────────────────")
    rc_f = rsa_v2["corr_concept"][last]
    rl_f = rsa_v2["corr_language_type"][last]
    rb_f = rsa_v2["corr_bias"][last]
    pc_f = probe_s["off_diag_mean"][last]
    chance = 0.2

    # Test 1: does concept beat language_type in RSA?
    t1 = rc_f > rl_f
    print(f"\n  Test 1 — RSA final layer: ρ(concept)={rc_f:+.4f} vs "
          f"ρ(lang_type)={rl_f:+.4f}")
    print(f"    → {'✔ concept > language_type' if t1 else '✘ language_type ≥ concept'}")

    # Test 2: cross-form probe above chance?
    t2 = pc_f > chance * 1.5
    print(f"\n  Test 2 — Cross-form probe={pc_f:.4f} (chance={chance:.2f})")
    print(f"    → {'✔ probe >> chance (concept info crosses form boundary)' if t2 else '✘ probe near chance'}")

    # Test 3: is β(same_language_type) smaller in magnitude than β(same_concept)?
    t3 = abs(bc) > abs(bl)
    print(f"\n  Test 3 — Regression β(concept)={bc:+.4f}  "
          f"β(lang_type)={bl:+.4f}  R²={r2:.4f}")
    print(f"    → {'✔ |β(concept)| > |β(lang_type)|  (concept, not language, drives distance)' if t3 else '✘ |β(lang_type)| ≥ |β(concept)|'}")

    # Overall
    n_support = sum([t1, t2, t3])
    print(f"\n  OVERALL: {n_support}/3 tests support RMR hypothesis", end="  ")
    if n_support == 3:
        print("→ FULL SUPPORT — strong pilot evidence")
    elif n_support == 2:
        print("→ PARTIAL SUPPORT — proceed with larger model/more stimuli")
    elif n_support == 1:
        print("→ WEAK SUPPORT — reconsider design or use larger model")
    else:
        print("→ NO SUPPORT — language-type dominates; refutes RMR in this setting")
    print()


def step_save_json(
    activations, rsa_v2, rsa_orig, probe_s, sil, cka_m,
    reg, crossover, bias_df, model_name, results_dir,
):
    """Save all numerical results to a structured JSON for future analysis."""
    N, L, D = activations.shape
    last = L - 1

    def arr(x):
        return [round(float(v), 6) for v in x]

    payload = {
        "metadata": {
            "model": model_name,
            "n_stimuli": N,
            "n_layers": L,
            "hidden_size": D,
            "run_date": datetime.now().isoformat(),
            "hypothesis": "RMR — Representation-Mediated Reasoning",
        },
        "per_layer": {
            # Extended RSA (4 RDMs)
            "rsa_concept":       arr(rsa_v2["corr_concept"]),
            "rsa_form":          arr(rsa_v2["corr_form"]),
            "rsa_bias":          arr(rsa_v2["corr_bias"]),
            "rsa_language_type": arr(rsa_v2["corr_language_type"]),
            "p_concept":         arr(rsa_v2["p_concept"]),
            "p_form":            arr(rsa_v2["p_form"]),
            "p_bias":            arr(rsa_v2["p_bias"]),
            "p_language_type":   arr(rsa_v2["p_language_type"]),
            # Silhouette
            "sil_concept": arr(sil["sil_concept"]),
            "sil_form":    arr(sil["sil_form"]),
            # Probe
            "probe_cross_form":  arr(probe_s["off_diag_mean"]),
            "probe_within_form": arr(probe_s["diagonal_mean"]),
            "probe_vs_chance":   arr(probe_s["cross_vs_chance"]),
            # CKA
            "cka_cross_form": arr(cka_m),
            # Regression betas
            "reg_beta_same_concept":       arr(reg["same_concept"]),
            "reg_beta_same_language_type": arr(reg["same_language_type"]),
            "reg_beta_token_diff":         arr(reg.get("token_count_diff",
                                                np.zeros(L))),
            "reg_beta_entropy_diff":       arr(reg.get("entropy_diff",
                                                np.zeros(L))),
            "reg_beta_ttr_diff":           arr(reg.get("ttr_diff",
                                                np.zeros(L))),
            "reg_r2": arr(reg["r2"]),
        },
        "scalar_summaries": {
            "crossover_layer":             crossover,
            "best_probe_cross_layer":      int(np.argmax(probe_s["off_diag_mean"])),
            "best_probe_cross_acc":        round(float(np.max(probe_s["off_diag_mean"])), 4),
            # Final layer values
            "final_layer_rsa_concept":     round(float(rsa_v2["corr_concept"][last]),       4),
            "final_layer_rsa_form":        round(float(rsa_v2["corr_form"][last]),           4),
            "final_layer_rsa_bias":        round(float(rsa_v2["corr_bias"][last]),           4),
            "final_layer_rsa_lang_type":   round(float(rsa_v2["corr_language_type"][last]),  4),
            "final_layer_sil_concept":     round(float(sil["sil_concept"][last]),            4),
            "final_layer_sil_form":        round(float(sil["sil_form"][last]),               4),
            "final_layer_cka_cross_form":  round(float(cka_m[last]),                         4),
            "final_layer_probe_cross":     round(float(probe_s["off_diag_mean"][last]),      4),
            "final_layer_reg_b_concept":   round(float(reg["same_concept"][last]),            4),
            "final_layer_reg_b_lang_type": round(float(reg["same_language_type"][last]),      4),
            "final_layer_reg_r2":          round(float(reg["r2"][last]),                      4),
            # RMR hypothesis verdict
            "rmr_test1_concept_gt_langtype": bool(
                rsa_v2["corr_concept"][last] > rsa_v2["corr_language_type"][last]),
            "rmr_test2_probe_above_chance":  bool(
                probe_s["off_diag_mean"][last] > 0.3),
            "rmr_test3_beta_concept_dominates": bool(
                abs(reg["same_concept"][last]) > abs(reg["same_language_type"][last])),
        },
        "bias_features_by_form": bias_df.groupby("form_name")[
            ["token_count", "token_entropy", "type_token_ratio", "mean_token_len"]
        ].mean().round(4).to_dict(),
        "regression_final_layer": {
            "beta_same_concept":       round(float(reg["same_concept"][last]),        4),
            "beta_same_language_type": round(float(reg["same_language_type"][last]),  4),
            "beta_token_count_diff":   round(float(reg.get("token_count_diff",
                                              np.zeros(L))[last]), 4),
            "beta_entropy_diff":       round(float(reg.get("entropy_diff",
                                              np.zeros(L))[last]), 4),
            "beta_ttr_diff":           round(float(reg.get("ttr_diff",
                                              np.zeros(L))[last]), 4),
            "r2":                      round(float(reg["r2"][last]), 4),
            "n_pairs":                 N * (N - 1) // 2,
        },
    }

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "pilot_numbers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Numerical results saved → {out_path}")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _model_slug(model_name: str) -> str:
    """Turn 'org/Model-Name-3B' into 'model-name-3b' for use in paths."""
    slug = model_name.split("/")[-1].lower()
    slug = slug.replace(" ", "_")
    return slug


def main():
    args = parse_args()
    model_name = args.model or config.MODEL_NAME

    # Default results dir is results/<model-slug>/ so each model is isolated
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(config.RESULTS_DIR, _model_slug(model_name))

    cache_path = os.path.join(results_dir, "activations_cache")

    stimuli, c_labels, f_labels = step_stimuli()
    activations = step_extract(stimuli, model_name, cache_path,
                               force=args.no_cache)
    bias_df, bias_rdm, lang_rdm, lang_labels, bias_matrix, bias_summary = \
        step_bias(stimuli, model_name)

    rsa_v2, rsa_orig, probe_t, probe_s, sil, cka_m, reg, crossover = \
        step_analyse(activations, c_labels, f_labels,
                     bias_rdm, lang_rdm, lang_labels, bias_matrix)

    step_summary(activations, rsa_v2, rsa_orig, probe_s,
                 sil, cka_m, reg, crossover, model_name)

    print("═" * 62)
    print("STEP 6 — Save numerical results to JSON")
    print("═" * 62)
    step_save_json(activations, rsa_v2, rsa_orig, probe_s, sil, cka_m,
                   reg, crossover, bias_df, model_name, results_dir)

    print("\n" + "═" * 62)
    print("STEP 7 — Generate figures")
    print("═" * 62)
    produce_all_figures_v2(
        activations     = activations,
        concept_labels  = c_labels,
        form_labels     = f_labels,
        stimuli_list    = stimuli,
        rsa_results     = rsa_orig,
        rsa_v2_results  = rsa_v2,
        sil_results     = sil,
        probe_summary   = probe_s,
        probe_transfer_tensor = probe_t,
        cka_mean        = cka_m,
        crossover_layer = crossover,
        bias_summary_df = bias_summary,
        reg_results     = reg,
        results_dir     = results_dir,
    )

    print("\nDone.  All results in:", os.path.abspath(results_dir))


if __name__ == "__main__":
    main()
