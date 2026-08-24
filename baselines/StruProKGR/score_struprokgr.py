#!/usr/bin/env python3
"""
Score StruProKGR outputs under the SparseKGC main protocol:
  filtered + full-entity ranking + bidirectional + tie-aware rank.

Dump format (written by StruProKGR.py --dump_scores_file):
  h TAB r TAB gold_t TAB gold_score TAB n_higher TAB n_tied

  n_higher: entities in filtered_answers with score > gold_score
  n_tied:   entities in filtered_answers with score == gold_score
            (includes gold if gold is predicted; 0 if gold_score=0 and gold is not predicted)

Full-entity tie-aware rank:
  If gold_score > 0 (gold was predicted and scored):
    rank = n_higher + (n_tied + 1) / 2

  If gold_score == 0 (gold not predicted or score=0):
    rank = n_higher_pos + (n_tied_zero + 1) / 2
    where:
      n_higher_pos = # predicted entities with score > 0 = n_predicted_nonzero (after filter)
      n_tied_zero  = (total_entities - n_filter) - n_predicted_nonzero
                   = all zero-score entities (includes unpredicted + gold + any predicted-but-zero)

Notes:
- n_higher and n_tied in the dump are computed AFTER gold-answer-specific filtering by StruProKGR.
- When gold_score == 0, n_higher is the count of entities with score > 0 in filtered_answers,
  and n_tied is the count with score == 0 in filtered_answers (predicted only).
- We extend n_tied_zero to include all unpredicted entities:
    total_zero = (total_entities - filter_size) - n_higher
    rank = n_higher + (total_zero + 1) / 2
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ranking_metrics import rank_from_counts  # noqa: E402


def load_sparsekgc_triples(data_dir: Path, dataset: str):
    """Build filter maps and entity set from SparseKGC-format data (h t r)."""
    tails_known = defaultdict(set)
    heads_known = defaultdict(set)
    entities = set()
    src = data_dir / dataset
    for split in ("train", "valid", "test"):
        p = src / f"{split}.txt"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                h, t, r = parts[0], parts[1], parts[2]
                tails_known[(h, r)].add(t)
                heads_known[(r, t)].add(h)
                entities.add(h)
                entities.add(t)
    return tails_known, heads_known, len(entities)


def load_dump(dump_path: Path):
    """Load a score dump into a list of (h, r, gold, score, n_higher, n_tied)."""
    rows = []
    with open(dump_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            h, r, gold, sc, nh, nt = parts
            rows.append((h, r, gold, float(sc), int(nh), int(nt)))
    return rows


def tie_aware_rank(gold_score, n_higher, n_tied_in_pred, filter_size, total_entities, tie_mode="average"):
    """Full-entity filtered rank of gold_answer.

    gold_score: StruProKGR's final score for the gold (0.0 if unpredicted).
      Scores are sums of precision * hop_factor terms (both non-negative: a
      precision is a hit-rate in [0, 1], hop_factor is a non-negative decay
      weight), so an unpredicted gold answer's true score is exactly 0 -- the
      same floor a predicted-but-zero-scored entity can also have. This is
      not an arbitrary default; it is the method's own definition of "no
      supporting path".
    n_higher: # predicted entities (after filter) with score > gold_score
    n_tied_in_pred: # predicted entities (after filter) with score == gold_score
    filter_size: # entities masked out (other known-true answers)
    total_entities: total entities in the KG
    tie_mode: "average" (Main Protocol) or "optimistic" (SOTA Protocol); both
      share this identical filtering and full-entity universe.
    """
    num_nonmasked = total_entities - filter_size
    if gold_score > 0.0:
        # gold is among predicted entities
        # n_tied_in_pred already includes gold itself
        return rank_from_counts(n_higher, n_tied_in_pred, tie_mode)
    else:
        # gold score = 0; all unpredicted entities also have score 0
        # n_higher = # predicted non-zero entities in filtered set
        # n_tied = total zero-score entities = non-masked - n_higher
        n_zero = num_nonmasked - n_higher
        return rank_from_counts(n_higher, n_zero, tie_mode)


def evaluate(forward_dump: Path, inverse_dump: Path, data_dir: Path, dataset: str, tie_mode: str = "average"):
    """Compute bidirectional filtered metrics.

    forward_dump: dump from running StruProKGR with test.triples (tail queries h r ?)
    inverse_dump: dump from running StruProKGR with test_inv.triples (head queries t r_inv ?)
    tie_mode: "average" (Main Protocol) or "optimistic" (SOTA Protocol).
    """
    tails_known, heads_known, n_ent = load_sparsekgc_triples(data_dir, dataset)

    agg = {d: {"mrr": 0.0, "h1": 0.0, "h3": 0.0, "h10": 0.0, "n": 0}
           for d in ("tail", "head")}

    # tail queries: forward dump
    for h, r, gold_t, gold_sc, n_higher, n_tied in load_dump(forward_dump):
        filter_set = tails_known.get((h, r), set()) - {gold_t}
        rank = tie_aware_rank(gold_sc, n_higher, n_tied, len(filter_set), n_ent, tie_mode=tie_mode)
        a = agg["tail"]
        a["mrr"] += 1.0 / rank
        a["h1"]  += 1.0 if rank <= 1 else 0.0
        a["h3"]  += 1.0 if rank <= 3 else 0.0
        a["h10"] += 1.0 if rank <= 10 else 0.0
        a["n"]   += 1

    # head queries: inverse dump (query is t r_inv ?, gold is h)
    def _orig_rel(r_inv: str) -> str:
        return r_inv[:-4] if r_inv.endswith("_inv") else r_inv + "_inv"

    if inverse_dump is not None:
        for t, r_inv, gold_h, gold_sc, n_higher, n_tied in load_dump(inverse_dump):
            orig_r = _orig_rel(r_inv)
            filter_set = heads_known.get((orig_r, t), set()) - {gold_h}
            rank = tie_aware_rank(gold_sc, n_higher, n_tied, len(filter_set), n_ent, tie_mode=tie_mode)
            a = agg["head"]
            a["mrr"] += 1.0 / rank
            a["h1"]  += 1.0 if rank <= 1 else 0.0
            a["h3"]  += 1.0 if rank <= 3 else 0.0
            a["h10"] += 1.0 if rank <= 10 else 0.0
            a["n"]   += 1

    res = {}
    for d in ("tail", "head"):
        a = agg[d]
        n = max(a["n"], 1)
        res[d] = {k: a[k] / n for k in ("mrr", "h1", "h3", "h10")}
    if agg["head"]["n"] > 0:
        res["avg"] = {k: (res["tail"][k] + res["head"][k]) / 2.0 for k in ("mrr", "h1", "h3", "h10")}
    else:
        res["avg"] = dict(res["tail"])
    res["tail_n"] = agg["tail"]["n"]
    res["head_n"] = agg["head"]["n"]
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--forward-dump", required=True)
    parser.add_argument("--inverse-dump", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    res = evaluate(Path(args.forward_dump), Path(args.inverse_dump),
                   Path(args.data_root), args.dataset)
    print(
        "FINAL_EVAL_METRICS baseline=struprokgr model=StruProKGR dataset={} split=test "
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
