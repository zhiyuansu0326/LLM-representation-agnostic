"""
Representation Bias Quantification for the RMR hypothesis.

For each stimulus we compute structural/informational properties of the
surface form that are independent of the semantic content:

  token_count         — verbosity / sparsity
  token_entropy       — Shannon H of token unigram distribution
  type_token_ratio    — lexical/symbol diversity
  mean_token_len      — morphological complexity (chars per token)

Language-type partition (for language-type RDM):
  natural_language : en_prose (form_id=0), zh_prose (form_id=3)
  formal/structured: py_code  (form_id=1), math     (form_id=2)

RMR operational prediction
──────────────────────────
In a regression that predicts pairwise representational distance:

  y[i,j] ~ β₀ + β₁·same_concept + β₂·same_language_type
             + β₃·|token_count_diff| + β₄·|entropy_diff| + β₅·|ttr_diff|

RMR predicts:
  |β₁| is the dominant predictor (concept drives similarity)
  |β₂| is NOT dominant after controlling for β₃…β₅

i.e., "is the pair across the nl/structured boundary?" does not
independently explain representational distance beyond structural bias.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd

# form_id mapping:  0=en_prose  1=py_code  2=math  3=zh_prose
_NL_FORM_IDS: set = {0, 3}


# ---------------------------------------------------------------------------
# Per-stimulus feature extraction
# ---------------------------------------------------------------------------

def _stats_from_ids(token_ids: List[int], decoded: List[str]) -> dict:
    n = len(token_ids)
    if n == 0:
        return {"token_count": 0, "token_entropy": 0.0,
                "type_token_ratio": 0.0, "mean_token_len": 0.0}
    counts = Counter(token_ids)
    total = sum(counts.values())
    entropy = -sum((c / total) * np.log2(c / total) for c in counts.values())
    ttr = len(counts) / n
    mean_len = float(np.mean([len(t.strip()) for t in decoded])) if decoded else 0.0
    return {"token_count": n, "token_entropy": entropy,
            "type_token_ratio": ttr, "mean_token_len": mean_len}


def _char_stats(text: str) -> dict:
    """Character-level fallback (used when tokenizer is unavailable)."""
    chars = list(text)
    n = len(chars)
    counts = Counter(chars)
    total = sum(counts.values())
    entropy = -sum((c / total) * np.log2(c / total) for c in counts.values()) if n > 1 else 0.0
    ttr = len(counts) / n if n > 0 else 0.0
    return {"token_count": n, "token_entropy": entropy,
            "type_token_ratio": ttr, "mean_token_len": 1.0}


def compute_bias_features(stimuli, tokenizer=None) -> pd.DataFrame:
    """
    Compute structural/informational bias features for all stimuli.

    Parameters
    ----------
    stimuli   : list of Stimulus objects
    tokenizer : HuggingFace tokenizer; falls back to char-level if None

    Returns
    -------
    pd.DataFrame  (one row per stimulus)
    Columns: concept_id, form_id, concept_name, form_name,
             is_natural_language, token_count, token_entropy,
             type_token_ratio, mean_token_len
    """
    rows = []
    for s in stimuli:
        if tokenizer is not None:
            ids = tokenizer.encode(s.text)
            decoded = [tokenizer.decode([t]) for t in ids]
            stats = _stats_from_ids(ids, decoded)
        else:
            stats = _char_stats(s.text)
        stats.update({
            "concept_id": s.concept_id,
            "form_id": s.form_id,
            "concept_name": s.concept_name,
            "form_name": s.form_name,
            "is_natural_language": int(s.form_id in _NL_FORM_IDS),
        })
        rows.append(stats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Theoretical RDMs derived from bias features
# ---------------------------------------------------------------------------

def build_bias_rdm(bias_df: pd.DataFrame) -> np.ndarray:
    """
    Build a pairwise RDM as the mean normalised L1 distance across three
    structural features: token_count, token_entropy, type_token_ratio.

    Interpretation: stimuli with similar structural properties have small
    bias distance.  If empirical RDM correlates with this RDM it means the
    model's representations are shaped by structural form properties.
    """
    cols = ["token_count", "token_entropy", "type_token_ratio"]
    X = bias_df[cols].values.astype(float)
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi - lo > 1e-10, hi - lo, 1.0)
    X_n = (X - lo) / span
    N = len(X_n)
    rdm = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            rdm[i, j] = float(np.mean(np.abs(X_n[i] - X_n[j])))
    return rdm


def build_language_type_rdm(stimuli) -> np.ndarray:
    """
    Binary RDM for language-type category:
      0 = same type  (both nl OR both structured)
      1 = cross-type (nl ↔ structured)

    This is the "language essentialism" RDM: a high empirical correlation
    with this RDM means the model treats natural language and structured
    forms as categorically distinct — which would *refute* RMR.
    """
    is_nl = np.array([int(s.form_id in _NL_FORM_IDS) for s in stimuli])
    return (is_nl[:, None] != is_nl[None, :]).astype(float)


# ---------------------------------------------------------------------------
# Helpers for regression
# ---------------------------------------------------------------------------

def get_language_type_labels(stimuli) -> List[int]:
    """0 = natural language, 1 = formal/structured."""
    return [int(s.form_id not in _NL_FORM_IDS) for s in stimuli]


def get_normalized_bias_matrix(bias_df: pd.DataFrame) -> np.ndarray:
    """
    Return (N, 3) normalized feature matrix for use in regression.
    Features: token_count, token_entropy, type_token_ratio (all in [0,1]).
    """
    cols = ["token_count", "token_entropy", "type_token_ratio"]
    X = bias_df[cols].values.astype(float)
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi - lo > 1e-10, hi - lo, 1.0)
    return (X - lo) / span


def bias_summary_by_form(bias_df: pd.DataFrame) -> pd.DataFrame:
    """Mean bias features grouped by form_name (for visualisation)."""
    return bias_df.groupby("form_name")[
        ["token_count", "token_entropy", "type_token_ratio", "mean_token_len"]
    ].mean()
