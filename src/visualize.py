"""
Visualisation module for the representation-agnostic pilot.

Produces publication-ready figures saved to a results/ directory:

  Fig 1 — RDM heatmaps at selected layers (empirical + two theoretical)
  Fig 2 — Layer-wise RSA curves: corr_concept vs. corr_form
  Fig 3 — Layer-wise silhouette curves: sil_concept vs. sil_form
  Fig 4 — Layer-wise off-diagonal cross-form probe accuracy
  Fig 5 — Layer-wise mean cross-form CKA
  Fig 6 — t-SNE of hidden states at selected layers, coloured by concept / by form
  Fig 7 — Cross-form probe transfer matrix heatmap (selected layer)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

from .stimuli import CONCEPT_NAMES, FORM_NAMES

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

CONCEPT_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
FORM_MARKERS   = ["o", "s", "^", "D"]
FORM_COLORS    = ["#264653", "#2A9D8F", "#E76F51", "#8338EC"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "lines.linewidth": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helper: cosine RDM
# ---------------------------------------------------------------------------

def _cosine_rdm(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    X_n = X / norms
    return cdist(X_n, X_n, metric="cosine")


def _tick_labels(stimuli_list) -> List[str]:
    return [f"{s.concept_name[:10]}\n{s.form_name[:6]}" for s in stimuli_list]


# ---------------------------------------------------------------------------
# Fig 1 — RDM heatmaps
# ---------------------------------------------------------------------------

def plot_rdm_heatmaps(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    stimuli_list,
    layers: Optional[List[int]] = None,
    save_path: str = "results/fig1_rdm_heatmaps.pdf",
) -> None:
    """
    Heatmaps of: conceptual theory | form theory | empirical (selected layers).
    """
    N, L, D = activations.shape
    if layers is None:
        layers = [0, L // 4, L // 2, 3 * L // 4, L - 1]

    n_theory = 2
    n_cols = n_theory + len(layers)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.2))

    tick_labels = _tick_labels(stimuli_list)

    # Theoretical RDMs
    def theory_rdm(labels):
        arr = np.array(labels)
        return (arr[:, None] != arr[None, :]).astype(float)

    for ax, (rdm, title) in zip(axes[:n_theory], [
        (theory_rdm(concept_labels), "Conceptual\n(theory)"),
        (theory_rdm(form_labels),    "Form\n(theory)"),
    ]):
        sns.heatmap(
            rdm, ax=ax, vmin=0, vmax=1,
            cmap="viridis", cbar=True,
            xticklabels=tick_labels, yticklabels=tick_labels,
            square=True,
        )
        ax.set_title(title)
        ax.tick_params(axis="both", labelsize=5, rotation=45)

    # Empirical RDMs
    for ax, layer in zip(axes[n_theory:], layers):
        rdm = _cosine_rdm(activations[:, layer, :])
        sns.heatmap(
            rdm, ax=ax, vmin=0, vmax=1,
            cmap="viridis", cbar=True,
            xticklabels=tick_labels, yticklabels=tick_labels,
            square=True,
        )
        ax.set_title(f"Layer {layer}\n(empirical)")
        ax.tick_params(axis="both", labelsize=5, rotation=45)

    fig.suptitle("Representational Dissimilarity Matrices", fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 2 — RSA curves
# ---------------------------------------------------------------------------

def plot_rsa_curves(
    rsa_results: Dict[str, np.ndarray],
    save_path: str = "results/fig2_rsa_curves.pdf",
    crossover_layer: Optional[int] = None,
) -> None:
    L = len(rsa_results["corr_concept"])
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, rsa_results["corr_concept"], color="#E63946", label="Conceptual RDM")
    ax.plot(layers, rsa_results["corr_form"],    color="#457B9D", label="Form RDM",
            linestyle="--")
    ax.fill_between(layers, rsa_results["corr_concept"], rsa_results["corr_form"],
                    where=rsa_results["corr_concept"] >= rsa_results["corr_form"],
                    alpha=0.15, color="#E63946", label="Concept > Form")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    if crossover_layer is not None and crossover_layer >= 0:
        ax.axvline(crossover_layer, color="black", linewidth=1.2, linestyle=":",
                   label=f"Crossover (layer {crossover_layer})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Spearman ρ with theoretical RDM")
    ax.set_title("RSA: Conceptual vs. Form structure across layers")
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 3 — Silhouette curves
# ---------------------------------------------------------------------------

def plot_silhouette_curves(
    sil_results: Dict[str, np.ndarray],
    save_path: str = "results/fig3_silhouette_curves.pdf",
    crossover_layer: Optional[int] = None,
) -> None:
    L = len(sil_results["sil_concept"])
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, sil_results["sil_concept"], color="#2A9D8F", label="Concept clustering")
    ax.plot(layers, sil_results["sil_form"],    color="#E9C46A", label="Form clustering",
            linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    if crossover_layer is not None and crossover_layer >= 0:
        ax.axvline(crossover_layer, color="black", linewidth=1.2, linestyle=":",
                   label=f"Crossover (layer {crossover_layer})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Silhouette coefficient (cosine)")
    ax.set_title("Clustering quality: Concept vs. Form labels across layers")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 4 — Cross-form probe accuracy
# ---------------------------------------------------------------------------

def plot_probe_curves(
    probe_summary: Dict[str, np.ndarray],
    save_path: str = "results/fig4_probe_transfer.pdf",
) -> None:
    L = len(probe_summary["off_diag_mean"])
    layers = np.arange(L)
    chance = 0.2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, probe_summary["diagonal_mean"], color="#264653",
            label="Within-form (same rep)")
    ax.plot(layers, probe_summary["off_diag_mean"],  color="#E76F51",
            label="Cross-form transfer")
    ax.axhline(chance, color="grey", linewidth=0.8, linestyle=":",
               label="Chance (20%)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Concept classification accuracy")
    ax.set_title("Linear probe: Within-form vs. Cross-form transfer accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 5 — CKA curves
# ---------------------------------------------------------------------------

def plot_cka_curves(
    cka_mean: np.ndarray,
    save_path: str = "results/fig5_cka_curves.pdf",
) -> None:
    L = len(cka_mean)
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, cka_mean, color="#8338EC", label="Mean cross-form CKA")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Cross-form representational alignment (CKA) across layers")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 6 — t-SNE scatter at selected layers
# ---------------------------------------------------------------------------

def plot_tsne(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    layers: Optional[List[int]] = None,
    save_path: str = "results/fig6_tsne.pdf",
    seed: int = 42,
) -> None:
    N, L, D = activations.shape
    if layers is None:
        layers = [0, L // 2, L - 1]

    n_layers = len(layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(4.5 * n_layers, 8))

    for col, layer in enumerate(layers):
        X = activations[:, layer, :]
        # Run t-SNE
        perplexity = min(5, N - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=2000)
        coords = tsne.fit_transform(X)

        # Row 0: colour by concept
        ax = axes[0, col]
        for cid in range(len(CONCEPT_NAMES)):
            mask = np.array(concept_labels) == cid
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=CONCEPT_COLORS[cid],
                marker=FORM_MARKERS[np.array(form_labels)[mask][0]],
                s=90, edgecolors="white", linewidths=0.5,
                label=CONCEPT_NAMES[cid][:12] if col == 0 else None,
                zorder=3,
            )
        ax.set_title(f"Layer {layer} — by Concept")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7,
                      markerscale=0.9)

        # Row 1: colour by form
        ax = axes[1, col]
        for fid in range(len(FORM_NAMES)):
            mask = np.array(form_labels) == fid
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=FORM_COLORS[fid],
                marker=FORM_MARKERS[fid],
                s=90, edgecolors="white", linewidths=0.5,
                label=FORM_NAMES[fid] if col == 0 else None,
                zorder=3,
            )
        ax.set_title(f"Layer {layer} — by Form")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7,
                      markerscale=0.9)

    fig.suptitle("t-SNE projections of hidden states", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 7 — Cross-form probe transfer matrix (selected layer)
# ---------------------------------------------------------------------------

def plot_transfer_matrix(
    transfer_tensor: np.ndarray,
    layer: int,
    save_path: str = "results/fig7_transfer_matrix.pdf",
) -> None:
    matrix = transfer_tensor[layer]
    n_forms = matrix.shape[0]
    form_labels_short = [f[:8] for f in FORM_NAMES[:n_forms]]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix, ax=ax, vmin=0, vmax=1, cmap="YlOrRd",
        annot=True, fmt=".2f", linewidths=0.5,
        xticklabels=[f"Test:\n{f}" for f in form_labels_short],
        yticklabels=[f"Train:\n{f}" for f in form_labels_short],
        square=True, cbar_kws={"label": "Accuracy"},
    )
    ax.set_title(f"Cross-form probe transfer matrix — Layer {layer}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Convenience: produce all figures at once
# ---------------------------------------------------------------------------

def produce_all_figures(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    stimuli_list,
    rsa_results: Dict,
    sil_results: Dict,
    probe_summary: Dict,
    probe_transfer_tensor: np.ndarray,
    cka_mean: np.ndarray,
    crossover_layer: int,
    results_dir: str = "results",
) -> None:
    N, L, D = activations.shape
    os.makedirs(results_dir, exist_ok=True)

    rdm_layers = sorted(set([0, L // 4, L // 2, 3 * L // 4, L - 1]))
    tsne_layers = sorted(set([0, L // 2, L - 1]))

    plot_rdm_heatmaps(
        activations, concept_labels, form_labels, stimuli_list,
        layers=rdm_layers,
        save_path=f"{results_dir}/fig1_rdm_heatmaps.pdf",
    )
    plot_rsa_curves(
        rsa_results,
        save_path=f"{results_dir}/fig2_rsa_curves.pdf",
        crossover_layer=crossover_layer,
    )
    plot_silhouette_curves(
        sil_results,
        save_path=f"{results_dir}/fig3_silhouette_curves.pdf",
        crossover_layer=crossover_layer,
    )
    plot_probe_curves(
        probe_summary,
        save_path=f"{results_dir}/fig4_probe_transfer.pdf",
    )
    plot_cka_curves(
        cka_mean,
        save_path=f"{results_dir}/fig5_cka_curves.pdf",
    )
    plot_tsne(
        activations, concept_labels, form_labels,
        layers=tsne_layers,
        save_path=f"{results_dir}/fig6_tsne.pdf",
    )
    best_layer = int(np.argmax(probe_summary["off_diag_mean"]))
    plot_transfer_matrix(
        probe_transfer_tensor, layer=best_layer,
        save_path=f"{results_dir}/fig7_transfer_matrix.pdf",
    )
    print(f"\nAll figures saved to {results_dir}/")


# ---------------------------------------------------------------------------
# Fig 8 — Bias features bar chart (per form)
# ---------------------------------------------------------------------------

def plot_bias_features_bar(
    bias_summary_df,
    save_path: str = "results/fig8_bias_features.pdf",
) -> None:
    """
    Grouped bar chart of structural bias features by representation form.
    Shows token_count, token_entropy, type_token_ratio side-by-side.
    """
    import pandas as pd

    features = ["token_count", "token_entropy", "type_token_ratio"]
    titles   = ["Token count", "Token entropy (bits)", "Type-token ratio"]

    # Normalise each feature to [0, 1] for comparable y-axes
    df = bias_summary_df[features].copy()
    for col in features:
        rng = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].min()) / rng if rng > 1e-10 else df[col]

    form_order  = ["en_prose", "py_code", "math", "zh_prose"]
    form_colors = dict(zip(form_order, FORM_COLORS))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, feat, title in zip(axes, features, titles):
        vals  = [df.loc[f, feat] if f in df.index else 0.0 for f in form_order]
        bars  = ax.bar(form_order, vals,
                       color=[form_colors.get(f, "#888") for f in form_order],
                       edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_xticklabels(form_order, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Normalised value")
        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Representation Bias Features by Form\n"
                 "(structural differences used as regression predictors)",
                 fontsize=11)
    # Language-type boundary annotation
    for ax in axes:
        ax.axvline(1.5, color="grey", linewidth=0.8, linestyle=":",
                   label="nl | structured boundary")
    axes[0].legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 9 — Extended RSA curves (4 theoretical RDMs)
# ---------------------------------------------------------------------------

def plot_rsa_curves_v2(
    rsa_v2_results: Dict,
    save_path: str = "results/fig9_rsa_v2_curves.pdf",
) -> None:
    """
    Four-curve RSA plot for RMR hypothesis.

    conceptual RDM  : does the model cluster by concept?
    form RDM        : does the model cluster by surface form?
    bias RDM        : does the model cluster by structural bias?
    language_type RDM: does the model cluster by nl vs structured?

    RMR prediction: conceptual curve should be the strongest in deep layers;
    language_type should not dominate.
    """
    L = len(rsa_v2_results["corr_concept"])
    layers = np.arange(L)

    styles = {
        "corr_concept":       ("#E63946", "-",  "Conceptual (same concept=0)"),
        "corr_form":          ("#457B9D", "--", "Surface form (same form=0)"),
        "corr_bias":          ("#2A9D8F", "-.", "Structural bias (feature dist)"),
        "corr_language_type": ("#E9C46A", ":",  "Language type (nl vs structured)"),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, (color, ls, label) in styles.items():
        ax.plot(layers, rsa_v2_results[key], color=color,
                linestyle=ls, linewidth=2, label=label)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

    # Shade region where concept > language_type (supports RMR)
    diff = rsa_v2_results["corr_concept"] - rsa_v2_results["corr_language_type"]
    ax.fill_between(layers, rsa_v2_results["corr_concept"],
                    rsa_v2_results["corr_language_type"],
                    where=diff > 0, alpha=0.12, color="#E63946",
                    label="concept > language_type (↑ RMR support)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Spearman ρ with theoretical RDM")
    ax.set_title("Extended RSA — 4 Theoretical RDMs (RMR hypothesis)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Fig 10 — Regression coefficient trajectories across layers
# ---------------------------------------------------------------------------

def plot_regression_curves(
    reg_results: Dict,
    save_path: str = "results/fig10_regression_betas.pdf",
) -> None:
    """
    Track standardised OLS betas across layers.

    RMR hypothesis predicts:
      β(same_concept) becomes the dominant negative predictor in deep layers
      β(same_language_type) does NOT independently dominate
    """
    L = len(reg_results["r2"])
    layers = np.arange(L)

    # Separate binary predictors from continuous bias diffs
    binary_preds = {
        "same_concept":       ("#E63946", "-",  "β: same concept"),
        "same_language_type": ("#E9C46A", "--", "β: same language type"),
    }
    bias_preds = {k: v for k, v in reg_results.items()
                  if k not in ("r2",) and k not in binary_preds}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: binary predictors (the core hypothesis test)
    for key, (color, ls, label) in binary_preds.items():
        if key in reg_results:
            ax1.plot(layers, reg_results[key], color=color,
                     linestyle=ls, linewidth=2.2, label=label)
    ax1.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax1.set_ylabel("Standardised β")
    ax1.set_title("RMR test: binary predictor betas across layers\n"
                  "(negative = same label → more similar representation)")
    ax1.legend(fontsize=9)
    # Annotate which dominates at final layer
    if "same_concept" in reg_results and "same_language_type" in reg_results:
        fc = reg_results["same_concept"][-1]
        fl = reg_results["same_language_type"][-1]
        winner = "concept" if abs(fc) > abs(fl) else "language_type"
        ax1.text(0.98, 0.05, f"Final layer: {winner} dominates",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=8, color="#333")

    # Bottom: continuous bias predictors + R²
    bias_colors = ["#2A9D8F", "#8338EC", "#F4A261"]
    for (key, arr), color in zip(
        {k: v for k, v in reg_results.items()
         if k not in ("r2", "same_concept", "same_language_type")}.items(),
        bias_colors,
    ):
        ax2.plot(layers, arr, color=color, linewidth=1.5, label=f"β: {key}")

    ax2r = ax2.twinx()
    ax2r.plot(layers, reg_results["r2"], color="black",
              linewidth=1.2, linestyle=":", label="R²")
    ax2r.set_ylabel("R²", fontsize=9)
    ax2r.set_ylim(0, 1.0)
    ax2r.legend(loc="upper right", fontsize=8)

    ax2.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Standardised β")
    ax2.set_title("Bias-feature betas (structural explanation) + model R²")
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Extended produce_all_figures
# ---------------------------------------------------------------------------

def produce_all_figures_v2(
    activations: np.ndarray,
    concept_labels,
    form_labels,
    stimuli_list,
    rsa_results: Dict,        # original 2-RDM RSA (kept for compatibility)
    rsa_v2_results: Dict,     # new 4-RDM RSA
    sil_results: Dict,
    probe_summary: Dict,
    probe_transfer_tensor: np.ndarray,
    cka_mean: np.ndarray,
    crossover_layer: int,
    bias_summary_df,
    reg_results: Dict,
    results_dir: str = "results",
) -> None:
    N, L, D = activations.shape
    os.makedirs(results_dir, exist_ok=True)

    rdm_layers  = sorted({0, L // 4, L // 2, 3 * L // 4, L - 1})
    tsne_layers = sorted({0, L // 2, L - 1})

    plot_rdm_heatmaps(
        activations, concept_labels, form_labels, stimuli_list,
        layers=rdm_layers,
        save_path=f"{results_dir}/fig1_rdm_heatmaps.pdf",
    )
    # Use the extended RSA as the primary RSA figure
    plot_rsa_curves_v2(
        rsa_v2_results,
        save_path=f"{results_dir}/fig2_rsa_v2_curves.pdf",
    )
    # Keep the original 2-curve RSA as supplementary
    plot_rsa_curves(
        rsa_results,
        save_path=f"{results_dir}/fig2b_rsa_original.pdf",
        crossover_layer=crossover_layer,
    )
    plot_silhouette_curves(
        sil_results,
        save_path=f"{results_dir}/fig3_silhouette_curves.pdf",
        crossover_layer=crossover_layer,
    )
    plot_probe_curves(
        probe_summary,
        save_path=f"{results_dir}/fig4_probe_transfer.pdf",
    )
    plot_cka_curves(
        cka_mean,
        save_path=f"{results_dir}/fig5_cka_curves.pdf",
    )
    plot_tsne(
        activations, concept_labels, form_labels,
        layers=tsne_layers,
        save_path=f"{results_dir}/fig6_tsne.pdf",
    )
    best_layer = int(np.argmax(probe_summary["off_diag_mean"]))
    plot_transfer_matrix(
        probe_transfer_tensor, layer=best_layer,
        save_path=f"{results_dir}/fig7_transfer_matrix.pdf",
    )
    plot_bias_features_bar(
        bias_summary_df,
        save_path=f"{results_dir}/fig8_bias_features.pdf",
    )
    plot_rsa_curves_v2(
        rsa_v2_results,
        save_path=f"{results_dir}/fig9_rsa_v2_curves.pdf",
    )
    plot_regression_curves(
        reg_results,
        save_path=f"{results_dir}/fig10_regression_betas.pdf",
    )
    print(f"\nAll figures saved to {results_dir}/")
