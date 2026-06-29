#!/usr/bin/env python3
"""Experiment artifact helpers for PathBSR.

The script is deliberately conservative:

* it reads only existing experiment outputs;
* it keeps validation, final test, current-default, and legacy runs separate;
* it marks missing ablation evidence instead of inventing numbers.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the PathBSR repository root")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))

from pathbsr import DEFAULT_CONFIG, PathBSR, load_dataset  # noqa: E402
from pathbsr.data import load_dataset_with_audit  # noqa: E402


DATASETS = ["FB15K-237-10", "FB15K-237-20", "FB15K-237-50", "NELL23K", "WD-singer"]
ALL_RESULT_DATASETS = DATASETS + ["WN18RR"]
SELECTED_STRUCTURAL_MODELS = ["TransE", "ConvE", "HoGRN", "PathBSR"]
WD_CAVEAT = "official split contains train/valid/test overlap caveat; no de-overlap run used"
MAIN_PROTOCOL = "validation; bidirectional filtered full-entity average-tie evaluation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(c, "")) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def pivot_metric(
    rows: list[dict[str, Any]],
    row_key: str,
    column_key: str,
    value_key: str,
    extra_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    extra_cols = extra_cols or []
    row_order: list[str] = []
    col_order: list[str] = []
    grouped: dict[str, dict[str, Any]] = {}
    extras: dict[str, dict[str, Any]] = {}
    for row in rows:
        rk = str(row.get(row_key, ""))
        ck = str(row.get(column_key, ""))
        if not rk or not ck:
            continue
        if rk not in row_order:
            row_order.append(rk)
        if ck not in col_order and ck not in {"ALL"}:
            col_order.append(ck)
        grouped.setdefault(rk, {})[ck] = row.get(value_key, "")
        extras.setdefault(rk, {col: row.get(col, "") for col in extra_cols})
    out: list[dict[str, Any]] = []
    for rk in row_order:
        item = {row_key: rk}
        item.update({col: grouped.get(rk, {}).get(col, "") for col in col_order})
        item.update(extras.get(rk, {}))
        out.append(item)
    return out


def fmt_float(value: Any, digits: int = 6) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def metric_row_by_dataset(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {row["dataset"]: row for row in rows if row.get("split") == "valid" or not row.get("split")}


def dataset_statistics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        train, valid, test, audit = load_dataset_with_audit(ROOT / "datasets", dataset)
        all_triples = train + valid + test
        entities = {e for h, _, t in all_triples for e in (h, t)}
        relations = {r for _, r, _ in all_triples}
        degree: dict[str, int] = defaultdict(int)
        undirected_edges = {tuple(sorted((h, t))) for h, _, t in train if h != t}
        for h, t in undirected_edges:
            degree[h] += 1
            degree[t] += 1
        avg_degree = (sum(degree.values()) / len(entities)) if entities else 0.0

        dup_parts = []
        for split in ("train", "valid", "test"):
            count = audit["splits"][split]["duplicate_count"]
            if count:
                dup_parts.append(f"{split}={count}")
        duplicate_note = "; ".join(dup_parts) if dup_parts else "none detected"

        overlap_parts = []
        for key in ("train_valid", "train_test", "valid_test"):
            count = audit["overlap"][key]["count"]
            if count:
                overlap_parts.append(f"{key}={count}")
        overlap_note = "; ".join(overlap_parts) if overlap_parts else "none detected"
        if dataset == "WD-singer":
            overlap_note = f"{overlap_note}; {WD_CAVEAT}"

        rows.append(
            {
                "Dataset": dataset,
                "#Entities": len(entities),
                "#Relations": len(relations),
                "#Train": len(train),
                "#Valid": len(valid),
                "#Test": len(test),
                "avg train degree": fmt_float(avg_degree, 3),
                "duplicate note": duplicate_note,
                "overlap note": overlap_note,
            }
        )
    return rows


















def reverse_relation(relation: str) -> str:
    suffix = DEFAULT_CONFIG.reverse_suffix
    if relation.endswith(suffix):
        return relation[: -len(suffix)]
    return f"{relation}{suffix}"


def clean_entity(entity: str) -> str:
    return entity.replace("concept_", "").replace("concept:", "")


def relation_programs_reaching_gold(model: PathBSR, h: str, r: str, gold: str, proxies: list[tuple[str, float]]) -> list[dict[str, Any]]:
    rk = model.ranker
    rm = model.rule_miner
    programs: list[dict[str, Any]] = []
    for path, _ in rm.get_relation_rules(r, model.config.rule_library_topk):
        weight = rm.path_answer_weight(r, path)
        if weight <= 0:
            continue
        outputs = rm.execute_path(h, path)
        if gold not in outputs:
            continue
        gated = any(rk._local_rule_support(proxy, r, path) > 0 for proxy, _ in proxies)
        programs.append(
            {
                "path": list(path),
                "reliability_weight": float(weight),
                "gate": "pass" if gated else "blocked",
                "num_query_outputs": len(outputs),
            }
        )
    programs.sort(key=lambda item: -float(item["reliability_weight"]))
    return programs[:8]


def trace_query(model: PathBSR, direction: str, original_h: str, original_r: str, original_t: str) -> dict[str, Any]:
    if direction == "tail":
        h, r, gold = original_h, original_r, original_t
    else:
        h, r, gold = original_t, reverse_relation(original_r), original_h
    rk = model.ranker
    g = model.graph
    proxies = rk.retriever.retrieve(h, r, model.config.topk_proxy)
    total_proxy = sum(score for _, score in proxies) or 1.0
    proxy_items = []
    proxy_gold_support = []
    for proxy, score in proxies[:10]:
        answers = sorted(g.out_adj.get(proxy, {}).get(r, set()))
        has_gold = gold in answers
        if has_gold:
            proxy_gold_support.append(proxy)
        proxy_items.append(
            {
                "proxy": proxy,
                "proxy_short": clean_entity(proxy),
                "normalized_bm25_weight": float(score / total_proxy),
                "has_gold_answer": has_gold,
                "num_relation_answers": len(answers),
                "sample_answers": [clean_entity(ans) for ans in answers[:5]],
            }
        )
    programs = relation_programs_reaching_gold(model, h, r, gold, proxies)
    case_scores = rk.score_case_paths(h, r, model.config.topk_proxy)
    candidate_scores = rk._apply_answer_base(case_scores, r)
    final_scores = rk.score(h, r, model.config.topk_proxy)
    prior = rk._answer_prior_prob(r)
    gold_idx = g.ent2idx[gold]
    rank = model.filtered_rank(final_scores, (h, r), gold)
    truth = g.all_true_tails.get((h, r), set())
    top_candidates = []
    for idx in np.argsort(-final_scores):
        candidate = g.all_entities[int(idx)]
        if candidate != gold and candidate in truth:
            continue
        top_candidates.append(
            {
                "entity": candidate,
                "entity_short": clean_entity(candidate),
                "final_score": float(final_scores[idx]),
                "candidate_score": float(candidate_scores[idx]),
                "local_path_proxy_score": float(case_scores[idx]),
                "frequency_score": float(prior[idx]),
                "verification_signal": float(rk.verification_value(h, candidate)),
                "is_gold": candidate == gold,
            }
        )
        if len(top_candidates) >= 5:
            break
    freq_top = []
    for idx in np.argsort(-prior)[:5]:
        if prior[idx] <= 0:
            break
        freq_top.append(
            {
                "entity": g.all_entities[int(idx)],
                "entity_short": clean_entity(g.all_entities[int(idx)]),
                "frequency_score": float(prior[idx]),
            }
        )
    return {
        "direction": direction,
        "query": {"head": h, "relation": r, "tail": "?"},
        "original_triple": {"head": original_h, "relation": original_r, "tail": original_t},
        "gold_answer": gold,
        "gold_answer_short": clean_entity(gold),
        "final_rank": float(rank),
        "gold_scores": {
            "final_score": float(final_scores[gold_idx]),
            "candidate_score": float(candidate_scores[gold_idx]),
            "local_path_proxy_score": float(case_scores[gold_idx]),
            "frequency_score": float(prior[gold_idx]),
            "verification_signal": float(rk.verification_value(h, gold)),
        },
        "top5_candidates": top_candidates,
        "retrieved_proxies": proxy_items,
        "proxy_gold_support_count": len(proxy_gold_support),
        "activated_relation_programs_for_gold": programs,
        "frequency_answers_top5": freq_top,
    }


def case_studies() -> list[dict[str, Any]]:
    train, valid, test = load_dataset(ROOT / "datasets", "NELL23K")
    model = PathBSR(train, valid, test, config=DEFAULT_CONFIG)
    found: dict[str, dict[str, Any]] = {}
    failure_candidate: dict[str, Any] | None = None

    for h, r, t in valid:
        for direction, oh, orl, ot in (("tail", h, r, t), ("head", h, r, t)):
            trace = trace_query(model, direction, oh, orl, ot)
            rank = float(trace["final_rank"])
            programs_pass = [
                item
                for item in trace["activated_relation_programs_for_gold"]
                if item["gate"] == "pass"
            ]
            proxy_support = int(trace["proxy_gold_support_count"])
            freq_score = float(trace["gold_scores"]["frequency_score"])

            if "successful_path_answer" not in found and rank <= 5 and programs_pass:
                trace["case_type"] = "successful Path Answer case"
                trace["short_explanation"] = (
                    "The gold answer is ranked highly and at least one gated relation program in the Global Path List "
                    "reaches the gold answer from the query head."
                )
                found["successful_path_answer"] = trace
            elif (
                "successful_proxy_frequency_fallback" not in found
                and rank <= 5
                and not programs_pass
                and (proxy_support > 0 or freq_score > 0)
            ):
                trace["case_type"] = "successful Proxy/Frequency fallback case"
                trace["short_explanation"] = (
                    "No gated path program reaches the gold answer, but proxy-answer support and/or the relation "
                    "frequency prior keeps the gold answer near the top."
                )
                found["successful_proxy_frequency_fallback"] = trace
            elif failure_candidate is None and rank > 100:
                trace["case_type"] = "failure case"
                trace["short_explanation"] = (
                    "The gold answer is ranked far below the top candidates; available proxy/path/frequency support "
                    "does not sufficiently distinguish it from competing candidates."
                )
                failure_candidate = trace

            if len(found) >= 2 and failure_candidate is not None:
                found["failure"] = failure_candidate
                return [
                    found["successful_path_answer"],
                    found["successful_proxy_frequency_fallback"],
                    found["failure"],
                ]

    if failure_candidate is not None:
        found["failure"] = failure_candidate
    return [found[key] for key in ("successful_path_answer", "successful_proxy_frequency_fallback", "failure") if key in found]


def write_case_markdown(path: Path, cases: list[dict[str, Any]]) -> None:
    lines = [
        "# PathBSR case studies",
        "",
        "Dataset: NELL23K validation split. Model: PathBSR default (`DEFAULT_CONFIG`, normalized BM25/top-100).",
        "",
    ]
    for idx, case in enumerate(cases, start=1):
        q = case["query"]
        lines.extend(
            [
                f"## Case {idx}: {case['case_type']}",
                "",
                f"- Query: `({clean_entity(q['head'])}, {q['relation']}, ?)`",
                f"- Gold answer: `{case['gold_answer_short']}`",
                f"- Final rank: `{case['final_rank']:.1f}`",
                f"- Gold scores: final={case['gold_scores']['final_score']:.6f}, "
                f"candidate={case['gold_scores']['candidate_score']:.6f}, "
                f"local={case['gold_scores']['local_path_proxy_score']:.6f}, "
                f"freq={case['gold_scores']['frequency_score']:.6f}, "
                f"verification={case['gold_scores']['verification_signal']:.6f}",
                f"- Explanation: {case['short_explanation']}",
                "",
                "Top-5 candidates:",
                "",
                "| rank | candidate | final | candidate | local | freq | verification | gold? |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for rank, candidate in enumerate(case["top5_candidates"], start=1):
            lines.append(
                f"| {rank} | `{candidate['entity_short']}` | {candidate['final_score']:.6f} | "
                f"{candidate['candidate_score']:.6f} | {candidate['local_path_proxy_score']:.6f} | "
                f"{candidate['frequency_score']:.6f} | {candidate['verification_signal']:.6f} | "
                f"{'yes' if candidate['is_gold'] else ''} |"
            )
        lines.extend(["", "Retrieved proxies:", ""])
        for proxy in case["retrieved_proxies"][:5]:
            lines.append(
                f"- `{proxy['proxy_short']}` weight={proxy['normalized_bm25_weight']:.4f}, "
                f"has_gold_answer={proxy['has_gold_answer']}, answers={proxy['sample_answers']}"
            )
        lines.extend(["", "Activated relation programs for gold:", ""])
        if case["activated_relation_programs_for_gold"]:
            for program in case["activated_relation_programs_for_gold"][:5]:
                lines.append(
                    f"- gate={program['gate']}, weight={program['reliability_weight']:.6f}, "
                    f"path=`{' -> '.join(program['path'])}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "Frequency Answers top-5:", ""])
        if case["frequency_answers_top5"]:
            for answer in case["frequency_answers_top5"]:
                lines.append(f"- `{answer['entity_short']}` freq={answer['frequency_score']:.6f}")
        else:
            lines.append("- none")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
