"""
Experiment configuration for the representation-agnostic pilot.

Edit MODEL_NAME to switch between a fast prototype model (gpt2)
and the main experiment model (gpt2-xl, or a LLaMA/Mistral variant).
"""

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis: Representation-Mediated Reasoning (RMR)
#
# "Reasoning capability in LLMs arises from operating over structured internal
#  representations, not from properties unique to natural language.  Natural
#  language is a privileged but non-essential representation."
#
# Operational prediction: in a regression predicting pairwise representational
# distance, β(same_language_type) should NOT dominate over β(same_concept)
# or β(bias_features) — language category is not an independent explanation.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

# Options (in ascending compute cost):
#   "gpt2"           124 M  — smoke-test, runs in seconds
#   "gpt2-medium"    355 M
#   "gpt2-large"     774 M
#   "gpt2-xl"       1558 M  — primary pilot model
#   "meta-llama/Llama-3.2-3B"  — stronger reasoning model (requires auth)
MODEL_NAME: str = "gpt2-xl"

# Use float16 on GPU to halve VRAM; fall back to float32 on CPU
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Where to cache the extracted activations (avoids re-running the model)
ACTIVATIONS_CACHE: str = "results/activations_cache.npz"

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

# Number of symbolic representation forms
N_FORMS: int = 4      # en_prose, py_code, math, zh_prose

# Number of abstract reasoning concepts
N_CONCEPTS: int = 5

# Random seed (for t-SNE, probe shuffles)
SEED: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR: str = "results"
