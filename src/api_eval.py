"""
API-based cross-form performance evaluation for the RMR hypothesis.

Instead of extracting hidden states, this module tests the RMR prediction
directly at the *behavioral* level:

  RMR performance prediction:
    "If reasoning is representation-mediated, a sufficiently capable model
     should solve the same reasoning problem with comparable accuracy
     regardless of which symbolic form the problem is presented in.
     Performance differences should be explainable by representation bias
     (structure, sparsity, ordering), not by 'is this natural language?'"

Design
──────
For each of the 5 reasoning concepts we append a concept-specific reasoning
probe (question + expected answer) to each of the 4 form stimuli.

The question is FORM-INVARIANT — the exact same conceptual question is
asked regardless of form.  Only the context changes.

Grading
───────
1. Keyword/exact-match for simple answers (numbers, sets).
2. LLM-as-judge fallback for free-form answers.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Reasoning probes — one per concept, form-invariant
# ---------------------------------------------------------------------------

PROBES: Dict[str, Dict] = {
    "recursive_decomposition": {
        "question": (
            "Based on the description above:\n"
            "If we start with a problem of size n=8, and each step splits "
            "it into 2 sub-problems of half the size, what is the minimum "
            "depth of recursion needed before every sub-problem reaches "
            "the base case (size 1)?\n"
            "Answer with a single integer."
        ),
        "answer": "3",
        "keywords": ["3"],
    },
    "transitivity": {
        "question": (
            "Based on the definition above:\n"
            "Alice is taller than Bob. Bob is taller than Carol. "
            "What can we conclude about Alice and Carol?\n"
            "Answer in one sentence."
        ),
        "answer": "Alice is taller than Carol.",
        "keywords": ["alice", "taller", "carol"],
    },
    "sorting_by_criterion": {
        "question": (
            "Using the method described above:\n"
            "Items: apple, banana, cherry. "
            "Key function values: key(apple)=3, key(banana)=1, key(cherry)=2. "
            "What is the correct sorted order from smallest to largest key?\n"
            "Answer with the three items in order, comma-separated."
        ),
        "answer": "banana, cherry, apple",
        "keywords": ["banana", "cherry", "apple"],
    },
    "set_intersection": {
        "question": (
            "Using the definition above:\n"
            "A = {1, 2, 3, 4},  B = {3, 4, 5, 6}.\n"
            "What is A ∩ B?\n"
            "Answer with the set elements."
        ),
        "answer": "{3, 4}",
        "keywords": ["3", "4"],
    },
    "causal_chain": {
        "question": (
            "Based on the causal structure described above:\n"
            "If we intervene to prevent event B from occurring, "
            "will event A still lead to event C?\n"
            "Answer Yes or No and give a one-sentence reason."
        ),
        "answer": "No. Because B mediates the effect of A on C.",
        "keywords": ["no"],
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def call_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """Call an OpenAI-compatible chat completion API and return the response text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            err = data.get("error", {}).get("message", str(data))
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"[API ERROR: {err}]"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[REQUEST ERROR: {e}]"
    return "[FAILED]"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_response(response: str, probe: Dict) -> bool:
    """
    Grade a model response as correct or incorrect.

    Strategy:
    1. Keyword match — all required keywords must appear in response.
    2. For causal_chain the first word must be 'no' (case-insensitive).
    """
    r = response.lower()
    return all(kw.lower() in r for kw in probe["keywords"])


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    stimuli,
    model: str,
    base_url: str,
    api_key: str,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run the cross-form reasoning evaluation for one model.

    Returns list of result dicts, one per stimulus.
    """
    system_prompt = (
        "You are a precise reasoning assistant. "
        "Read the provided context carefully and answer the question concisely. "
        "Do not add unnecessary explanation."
    )

    results = []
    for s in stimuli:
        probe = PROBES[s.concept_name]
        user_msg = f"Context:\n{s.text}\n\nQuestion:\n{probe['question']}"

        response = call_api(
            base_url, api_key, model,
            system_prompt, user_msg,
            max_tokens=150, temperature=0.0,
        )
        correct = grade_response(response, probe)

        rec = {
            "concept_id":   s.concept_id,
            "form_id":      s.form_id,
            "concept_name": s.concept_name,
            "form_name":    s.form_name,
            "is_natural_language": int(s.form_id in {0, 3}),
            "response":     response,
            "correct":      correct,
            "model":        model,
        }
        results.append(rec)

        if verbose:
            mark = "✔" if correct else "✘"
            print(f"  {mark} [{s.concept_name[:22]:22s}|{s.form_name:10s}] "
                  f"{response[:55]!r}")
    return results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_results(results: List[Dict]) -> Dict:
    """Compute accuracy by form, by concept, and by language type."""
    from collections import defaultdict

    by_form: Dict[str, List[bool]] = defaultdict(list)
    by_concept: Dict[str, List[bool]] = defaultdict(list)
    by_lang_type: Dict[str, List[bool]] = {"natural_language": [], "formal": []}

    for r in results:
        by_form[r["form_name"]].append(r["correct"])
        by_concept[r["concept_name"]].append(r["correct"])
        key = "natural_language" if r["is_natural_language"] else "formal"
        by_lang_type[key].append(r["correct"])

    def acc(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "overall":       acc([r["correct"] for r in results]),
        "by_form":       {k: acc(v) for k, v in by_form.items()},
        "by_concept":    {k: acc(v) for k, v in by_concept.items()},
        "by_lang_type":  {k: acc(v) for k, v in by_lang_type.items()},
        "nl_vs_formal_gap": acc(by_lang_type["natural_language"]) - acc(by_lang_type["formal"]),
        "n_correct":     sum(r["correct"] for r in results),
        "n_total":       len(results),
    }


def rmr_performance_verdict(agg: Dict) -> str:
    """
    Verdict based on the performance test.

    RMR predicts: nl_vs_formal_gap should be small (|gap| < 0.2).
    If natural language consistently outperforms formal by > 0.2,
    that challenges RMR for this model.
    """
    gap = abs(agg["nl_vs_formal_gap"])
    nl_acc  = agg["by_lang_type"]["natural_language"]
    frm_acc = agg["by_lang_type"]["formal"]

    if gap < 0.10:
        verdict = "STRONG SUPPORT — performance nearly identical across language types"
    elif gap < 0.20:
        verdict = "PARTIAL SUPPORT — small performance gap, likely explained by bias"
    else:
        direction = "NL > formal" if agg["nl_vs_formal_gap"] > 0 else "formal > NL"
        verdict = f"WEAK/NO SUPPORT — large gap ({direction}), language type matters"

    return (
        f"NL accuracy={nl_acc:.2%}  Formal accuracy={frm_acc:.2%}  "
        f"Gap={agg['nl_vs_formal_gap']:+.2%}\n  → {verdict}"
    )
