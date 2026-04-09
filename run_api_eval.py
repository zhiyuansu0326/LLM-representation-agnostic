"""
run_api_eval.py — Cross-form reasoning performance test via external API.

Tests the RMR hypothesis at the BEHAVIORAL level:

  For each of 20 stimuli (5 concepts × 4 forms), presents the stimulus
  + a form-invariant reasoning question to the model via API, then grades
  correctness.

  RMR prediction: accuracy should be comparable across nl and formal forms;
  any performance gap should correlate with structural bias features
  (token count, entropy, TTR), not with the binary is_natural_language flag.

Usage
─────
  python run_api_eval.py \\
      --api-base http://14.103.68.46 \\
      --api-key  sk-xxx \\
      --models   gpt-5 claude-opus-4-5 qwen3-235b-a22b

  Results saved to: results/api_eval/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stimuli import build_stimuli
from src.api_eval import evaluate_model, aggregate_results, rmr_performance_verdict
from src.rep_bias import compute_bias_features


RESULTS_DIR = "results/api_eval"

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-base",  default="http://14.103.68.46")
    p.add_argument("--api-key",   required=True)
    p.add_argument("--models",    nargs="+",
                   default=["gpt-5", "claude-opus-4-5", "qwen3-235b-a22b"])
    p.add_argument("--results-dir", default=RESULTS_DIR)
    return p.parse_args()


def print_model_summary(model: str, agg: dict, verdict: str):
    sep = "─" * 60
    print(f"\n  Model: {model}")
    print(f"  {sep}")
    print(f"  Overall accuracy: {agg['overall']:.2%}  "
          f"({agg['n_correct']}/{agg['n_total']})")
    print(f"\n  By form:")
    for form, acc in sorted(agg["by_form"].items()):
        lang = "(NL)  " if form in ("en_prose","zh_prose") else "(code)"
        print(f"    {form:12s} {lang}  {acc:.2%}")
    print(f"\n  By concept:")
    for concept, acc in sorted(agg["by_concept"].items()):
        print(f"    {concept:28s}  {acc:.2%}")
    print(f"\n  RMR Performance Test:")
    print(f"    {verdict}")
    print(f"  {sep}")


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    stimuli = build_stimuli()

    # Compute bias features (char-level fallback, no tokenizer needed here)
    bias_df = compute_bias_features(stimuli, tokenizer=None)

    all_model_results = {}
    comparison_table = []

    for model in args.models:
        print(f"\n{'═'*62}")
        print(f"  Model: {model}")
        print(f"{'═'*62}")

        results = evaluate_model(
            stimuli, model, args.api_base, args.api_key, verbose=True
        )
        agg     = aggregate_results(results)
        verdict = rmr_performance_verdict(agg)

        print_model_summary(model, agg, verdict)
        all_model_results[model] = {"results": results, "aggregate": agg}

        comparison_table.append({
            "model":          model,
            "overall_acc":    agg["overall"],
            "nl_acc":         agg["by_lang_type"]["natural_language"],
            "formal_acc":     agg["by_lang_type"]["formal"],
            "nl_formal_gap":  round(agg["nl_vs_formal_gap"], 4),
            "by_form":        agg["by_form"],
            "by_concept":     agg["by_concept"],
        })

    # ── Cross-model summary ──────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("  CROSS-MODEL RMR PERFORMANCE SUMMARY")
    print(f"{'═'*62}")
    print(f"\n  {'Model':<30} {'Overall':>8} {'NL acc':>8} {'Formal':>8} {'Gap':>8}")
    print(f"  {'─'*62}")
    for row in comparison_table:
        print(f"  {row['model']:<30} "
              f"{row['overall_acc']:>7.1%}  "
              f"{row['nl_acc']:>7.1%}  "
              f"{row['formal_acc']:>7.1%}  "
              f"{row['nl_formal_gap']:>+7.1%}")

    # RMR verdict: is the gap consistent with representation bias?
    print(f"\n  RMR Prediction: |NL−Formal gap| < 20% across all models")
    for row in comparison_table:
        gap = abs(row["nl_formal_gap"])
        mark = "✔" if gap < 0.20 else "✘"
        print(f"  {mark}  {row['model']}: gap={row['nl_formal_gap']:+.1%}")

    # ── Save JSON ────────────────────────────────────────────────────────
    payload = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "api_base": args.api_base,
            "models":   args.models,
            "n_stimuli": len(stimuli),
            "hypothesis": "RMR — Representation-Mediated Reasoning",
            "test": "cross-form reasoning performance consistency",
        },
        "comparison": comparison_table,
        "per_model_details": {
            model: data["aggregate"]
            for model, data in all_model_results.items()
        },
        "per_stimulus_responses": {
            model: [
                {k: v for k, v in r.items() if k != "response"}  # omit raw text
                for r in data["results"]
            ]
            for model, data in all_model_results.items()
        },
        "per_stimulus_responses_full": {
            model: data["results"]
            for model, data in all_model_results.items()
        },
    }

    out = os.path.join(args.results_dir, "api_eval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()
