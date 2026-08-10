#!/usr/bin/env python3
"""
Score an AnyBURL prediction file under the SAME evaluation protocol used by the
other SparseKGC baselines (filtered, tie-aware, bidirectional, full-entity
ranking), so AnyBURL is apples-to-apples comparable.

AnyBURL `Apply` writes, per test triple, three lines:

    <h> <r> <t>
    Heads: e1 <TAB> s1 <TAB> e2 <TAB> s2 ...     (candidates for (?, r, t))
    Tails: e1 <TAB> s1 <TAB> e2 <TAB> s2 ...     (candidates for (h, r, ?))

Ranking (matching baselines/traditional/main.py):
  - Filtered: other known-true entities (from train+valid+test) are removed.
  - Tie-aware: rank = #{score > target} + (#{score == target} + 1) / 2,
    computed over all N non-masked entities.
  - Entities AnyBURL did not predict have score 0. When the target is among
    them (no rule fired), the same tie-aware formula assigns it the average
    rank over all 0-score entities -- the consistent analogue of how main.py
    would rank a 0 score, rather than an ad-hoc "rank = N".

`Tails:` -> tail metrics (target t), `Heads:` -> head metrics (target h),
matching main.py's tail / inverse-relation-head split.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from metrics_csv import upsert_metrics_csv  # noqa: E402


def load_triples_hrt(path):
    """Read AnyBURL-format (h r t) triples."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            out.append((h, r, t))
    return out


def build_filters(work_dir):
    tails_known = defaultdict(set)   # (h, r) -> {t}
    heads_known = defaultdict(set)   # (r, t) -> {h}
    entities = set()
    for split in ("train", "valid", "test"):
        p = Path(work_dir) / f"{split}.txt"
        if not p.exists():
            continue
        for h, r, t in load_triples_hrt(p):
            tails_known[(h, r)].add(t)
            heads_known[(r, t)].add(h)
            entities.add(h)
            entities.add(t)
    return tails_known, heads_known, len(entities)


def parse_predictions(pred_path):
    """Yield (h, r, t, heads_list, tails_list); each list is [(entity, score)]."""
    def parse_cand(line, prefix):
        # After the "Heads:"/"Tails:" prefix there is a leading space before the
        # first entity; strip it (and each token) so entity ids match exactly.
        body = line[len(prefix):].strip()
        toks = [tok.strip() for tok in body.split("\t") if tok.strip() != ""]
        cands = []
        for i in range(0, len(toks) - 1, 2):
            ent = toks[i]
            try:
                sc = float(toks[i + 1])
            except ValueError:
                continue
            cands.append((ent, sc))
        return cands

    with open(pred_path) as f:
        lines = f.read().split("\n")
    i = 0
    while i + 2 < len(lines) + 1:
        if i >= len(lines):
            break
        triple_line = lines[i]
        if triple_line.strip() == "":
            i += 1
            continue
        parts = triple_line.split(" ")
        if len(parts) < 3:
            i += 1
            continue
        h, r, t = parts[0], parts[1], parts[2]
        heads_list = parse_cand(lines[i + 1], "Heads:") if i + 1 < len(lines) and lines[i + 1].startswith("Heads:") else []
        tails_list = parse_cand(lines[i + 2], "Tails:") if i + 2 < len(lines) and lines[i + 2].startswith("Tails:") else []
        yield h, r, t, heads_list, tails_list
        i += 3


def tie_aware_rank(cands, target, filtered_others, num_entities):
    """Filtered tie-aware rank of `target` given AnyBURL candidate (entity,score).

    Entities not in `cands` have score 0. `filtered_others` (known-true entities
    other than the target) are masked out of the ranking entirely.
    """
    # candidate score map, excluding masked entities; dedupe keeping max score
    score = {}
    for ent, sc in cands:
        if ent in filtered_others:
            continue
        if ent not in score or sc > score[ent]:
            score[ent] = sc
    target_score = score.get(target, 0.0)

    # non-masked entity count: all entities minus the masked others.
    # (target itself is never in filtered_others.)
    num_nonmasked = num_entities - len(filtered_others)

    if target_score > 0.0:
        greater = sum(1 for e, s in score.items() if s > target_score)
        equal = sum(1 for e, s in score.items() if s == target_score)  # includes target
        return greater + (equal + 1.0) / 2.0
    else:
        # target unpredicted (score 0). All positive-score candidates beat it.
        pos = sum(1 for s in score.values() if s > 0.0)
        greater = pos
        # entities tied at score 0 = non-masked entities that are not positive
        # candidates (includes the target).
        equal = num_nonmasked - pos
        return greater + (equal + 1.0) / 2.0


def evaluate(pred_path, work_dir):
    tails_known, heads_known, num_ent = build_filters(work_dir)

    agg = {d: {"mrr": 0.0, "h1": 0, "h3": 0, "h10": 0, "n": 0} for d in ("tail", "head")}

    for h, r, t, heads_list, tails_list in parse_predictions(pred_path):
        # tail prediction: query (h, r, ?), target t
        filt_t = tails_known.get((h, r), set()) - {t}
        rank_t = tie_aware_rank(tails_list, t, filt_t, num_ent)
        # head prediction: query (?, r, t), target h
        filt_h = heads_known.get((r, t), set()) - {h}
        rank_h = tie_aware_rank(heads_list, h, filt_h, num_ent)

        for d, rank in (("tail", rank_t), ("head", rank_h)):
            a = agg[d]
            a["mrr"] += 1.0 / rank
            a["h1"] += 1 if rank <= 1 else 0
            a["h3"] += 1 if rank <= 3 else 0
            a["h10"] += 1 if rank <= 10 else 0
            a["n"] += 1

    res = {}
    for d in ("tail", "head"):
        a = agg[d]
        n = max(a["n"], 1)
        res[d] = {
            "mrr": a["mrr"] / n,
            "h1": a["h1"] / n,
            "h3": a["h3"] / n,
            "h10": a["h10"] / n,
        }
    res["avg"] = {k: (res["tail"][k] + res["head"][k]) / 2.0 for k in ("mrr", "h1", "h3", "h10")}
    return res


def main():
    parser = argparse.ArgumentParser(description="Score AnyBURL predictions under the SparseKGC protocol")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--work-dir", required=True, help="dir with AnyBURL-format train/valid/test.txt")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--metrics-csv", default=None)
    args = parser.parse_args()

    res = evaluate(args.predictions, args.work_dir)
    print(
        "FINAL_EVAL_METRICS baseline=anyburl model=AnyBURL dataset={} split=test "
        "mrr_tail={:.5f} mrr_head={:.5f} mrr_avg={:.5f} "
        "h1_tail={:.5f} h1_head={:.5f} h1_avg={:.5f} "
        "h3_tail={:.5f} h3_head={:.5f} h3_avg={:.5f} "
        "h10_tail={:.5f} h10_head={:.5f} h10_avg={:.5f}".format(
            args.dataset,
            res["tail"]["mrr"], res["head"]["mrr"], res["avg"]["mrr"],
            res["tail"]["h1"], res["head"]["h1"], res["avg"]["h1"],
            res["tail"]["h3"], res["head"]["h3"], res["avg"]["h3"],
            res["tail"]["h10"], res["head"]["h10"], res["avg"]["h10"],
        )
    )

    if args.metrics_csv:
        upsert_metrics_csv(args.metrics_csv, [
            args.dataset, "AnyBURL",
            f"{res['tail']['mrr']:.5f}", f"{res['head']['mrr']:.5f}", f"{res['avg']['mrr']:.5f}",
            f"{res['tail']['h1']:.5f}", f"{res['head']['h1']:.5f}", f"{res['avg']['h1']:.5f}",
            f"{res['tail']['h3']:.5f}", f"{res['head']['h3']:.5f}", f"{res['avg']['h3']:.5f}",
            f"{res['tail']['h10']:.5f}", f"{res['head']['h10']:.5f}", f"{res['avg']['h10']:.5f}",
            f"{args.seconds:.3f}",
        ])


if __name__ == "__main__":
    main()
