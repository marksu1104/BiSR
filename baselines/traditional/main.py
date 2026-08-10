import argparse
import csv
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models.kge_models import TransE, DistMult
from data.dataloader import KGData, TrainDataset, TrainDatasetOriginal, EvalDataset, HeadEvalDataset, get_dataloader

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
        
        # Load Data WITH Inverse Relations (Matching HoGRN protocol)
        # This doubles the relations and allows Head prediction via Tail prediction on inverse triples
        print(f"Loading data from {args.data_path}...")
        self.kg_data = KGData(args.data_path, add_inverse=bool(args.add_inverse))
        self.args.num_ent = self.kg_data.num_ent
        self.args.num_rel = self.kg_data.num_rel
        
        print(f"Dataset Loaded. Num Ent: {self.args.num_ent}, Num Rel: {self.args.num_rel}")
        
        # Build Model
        self.model = self._build_model().to(self.device)
        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)

        # Training regime:
        #   "bce"    -> 1-vs-All BCE (HoGRN protocol; used by ConvE/TuckER/DistMult/ComplEx/RotatE)
        #   "margin" -> classic pairwise margin-ranking with negative sampling
        #               (faithful to the original TransE; Bordes et al. 2013)
        self.loss_mode = getattr(args, "loss", "bce")

        # Checkpoints dir
        self.ckpt_dir = os.environ.get(
            "SPARSEKGC_CHECKPOINT_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Checkpoint filename. A --run_tag (default "") keeps the legacy name so
        # --eval_only still finds it; concurrent runs of the same model+dataset
        # must pass distinct tags to avoid overwriting each other's best model.
        run_tag = getattr(args, "run_tag", "") or ""
        suffix = f"_{run_tag}" if run_tag else ""
        self.ckpt_path = os.path.join(
            self.ckpt_dir, f"best_model_{args.model}_{args.dataset}{suffix}.pth"
        )

        if self.loss_mode == "margin":
            # Classic training: pairwise (positive, corrupted-negative) triples.
            self.criterion = nn.MarginRankingLoss(margin=args.margin_rank)
            train_dataset = TrainDatasetOriginal(self.kg_data.train_triples, self.args.num_ent)
        else:
            # 1-vs-All BCE over all entities.
            self.criterion = nn.BCELoss()
            train_dataset = TrainDataset(self.kg_data.train_sr2o, self.args.num_ent,
                                         ls_dackgr=bool(getattr(args, "ls_dackgr", 0)))
        self.train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        
    def _build_model(self):
        if self.args.model == 'TransE':
            return TransE(self.args)
        elif self.args.model == 'DistMult':
            return DistMult(self.args)
        elif self.args.model == 'RotatE':
            # RotatE in kge_models.py but not imported or defined in models list previously?
            # It was in the file, we need to import it.
            # Let's assume it's imported (I will update imports below if needed)
            from models.kge_models import RotatE 
            return RotatE(self.args)
        elif self.args.model == 'ComplEx':
            from models.kge_models import ComplEx
            return ComplEx(self.args)
        elif self.args.model == 'ConvE':
            from models.kge_models import ConvE
            return ConvE(self.args)
        elif self.args.model == 'TuckER':
            from models.kge_models import TuckER
            return TuckER(self.args)
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

    def train(self):
        best_mrr = 0.0
        kill_cnt = 0
        
        for epoch in range(1, self.args.max_epochs + 1):
            # print("########")
            t0 = time.time()
            
            self.model.train()
            total_loss = 0

            if self.loss_mode == "margin":
                total_loss = self._train_epoch_margin()
            else:
                # Using 1-vs-All DataLoader
                # batch: (inputs, labels)
                # inputs: (Batch, 2) -> [h, r]
                # labels: (Batch, NumEnt) -> Multi-hot
                for inputs, labels in self.train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    h, r = inputs[:, 0], inputs[:, 1]

                    # Forward (Returns Sigmoid Scores)
                    # (Batch, NumEnt)
                    preds = self.model(h, r)

                    loss = self.criterion(preds, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            # print("Time cost in one epoch for training: {:.4f}s".format((time.time()-t0)))
            
            # Evaluation
            if epoch % self.args.eval_freq == 0:
                if self.args.selection_protocol == "sota":
                    val_metrics = self.evaluate_sota(split="valid", epoch=epoch)
                else:
                    val_metrics = self.evaluate(split="valid", epoch=epoch)
                current_mrr = val_metrics["mrr"]
                
                if current_mrr > best_mrr:
                    best_mrr = current_mrr
                    kill_cnt = 0
                    torch.save(self.model.state_dict(), self.ckpt_path)
                else:
                    kill_cnt += 1
                    if kill_cnt >= self.args.patience:
                        print("Early Stopping!!")
                        print('[Epoch {}]: Training Loss: {:.5}, Best Valid MRR: {:.5}\n\n'.format(epoch, avg_loss, best_mrr))
                        break
                
                print('[Epoch {}]: Training Loss: {:.5}, Best Valid MRR: {:.5}\n\n'.format(epoch, avg_loss, best_mrr))
            else:
                print('[Epoch {}]: Training Loss: {:.5}\n'.format(epoch, avg_loss))

    def _train_epoch_margin(self):
        """Classic pairwise margin-ranking epoch with negative sampling.

        Each batch item is (positive_triple, corrupted_triple). We enforce the
        original TransE constraint ||e||_2 <= 1 on entity embeddings, score both
        triples with the model's pairwise mode (higher score = better), and apply
        a margin-ranking loss so the positive outscores the negative by `margin`.
        """
        total_loss = 0.0
        for pos, neg in self.train_loader:
            pos = pos.to(self.device)
            neg = neg.to(self.device)

            # Classic TransE: constrain entity embeddings to the unit ball.
            self.model.ent_emb.weight.data.renorm_(p=2, dim=0, maxnorm=1.0)

            pos_score = self.model(pos[:, 0], pos[:, 1], pos[:, 2])
            neg_score = self.model(neg[:, 0], neg[:, 1], neg[:, 2])

            target = torch.ones_like(pos_score)
            loss = self.criterion(pos_score, neg_score, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss

    def _run_eval_pass(self, eval_triples):
        """Return main tie-aware and SOTA optimistic filtered metrics."""
        totals = {mode: {key: 0.0 for key in ("mrr", "h1", "h3", "h10")}
                  for mode in ("tie", "optimistic")}
        total = 0
        eval_dataset = EvalDataset(eval_triples, self.kg_data.all_sr2o, self.args.num_ent)
        data_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size,
                                 shuffle=False, num_workers=self.args.num_workers)

        with torch.no_grad():
            for triples, labels in data_loader:
                triples = triples.to(self.device)
                labels = labels.to(self.device)
                h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
                scores = self.model(h, r)
                b_range = torch.arange(scores.size(0), device=self.device)
                target_score = scores[b_range, t]
                scores = scores.masked_fill(labels.bool(), -float("inf"))
                scores[b_range, t] = target_score

                greater = (scores > target_score.unsqueeze(1)).sum(dim=1).float()
                equal = (scores == target_score.unsqueeze(1)).sum(dim=1).float()
                ranks = {
                    "tie": greater + (equal + 1.0) / 2.0,
                    "optimistic": greater + 1.0,
                }
                for mode, rank in ranks.items():
                    totals[mode]["mrr"] += (1.0 / rank).sum().item()
                    totals[mode]["h1"] += (rank <= 1).sum().item()
                    totals[mode]["h3"] += (rank <= 3).sum().item()
                    totals[mode]["h10"] += (rank <= 10).sum().item()
                total += scores.size(0)

        return {mode: {key: value / total for key, value in values.items()}
                for mode, values in totals.items()}

    def _run_head_eval_pass(self, eval_triples):
        """Direct filtered head ranking for models trained without reciprocals."""
        totals = {mode: {key: 0.0 for key in ("mrr", "h1", "h3", "h10")}
                  for mode in ("tie", "optimistic")}
        total = 0
        dataset = HeadEvalDataset(eval_triples, self.kg_data.all_ro2s, self.args.num_ent)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                            num_workers=self.args.num_workers)
        with torch.no_grad():
            for triples, labels in loader:
                triples = triples.to(self.device)
                labels = labels.to(self.device)
                h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
                scores = self.model.score_heads(r, t)
                b_range = torch.arange(scores.size(0), device=self.device)
                target_score = scores[b_range, h]
                scores = scores.masked_fill(labels.bool(), -float("inf"))
                scores[b_range, h] = target_score
                greater = (scores > target_score.unsqueeze(1)).sum(dim=1).float()
                equal = (scores == target_score.unsqueeze(1)).sum(dim=1).float()
                ranks = {"tie": greater + (equal + 1.0) / 2.0,
                         "optimistic": greater + 1.0}
                for mode, rank in ranks.items():
                    totals[mode]["mrr"] += (1.0 / rank).sum().item()
                    totals[mode]["h1"] += (rank <= 1).sum().item()
                    totals[mode]["h3"] += (rank <= 3).sum().item()
                    totals[mode]["h10"] += (rank <= 10).sum().item()
                total += scores.size(0)
        return {mode: {key: value / total for key, value in values.items()}
                for mode, values in totals.items()}

    def evaluate_sota(self, split="valid", epoch=0, label=None):
        """Paper comparison: tail-only, filtered, optimistic tie handling."""
        self.model.eval()
        queries = list(getattr(self.kg_data, f"{split}_triples"))
        results = self._run_eval_pass(queries)["optimistic"]
        eval_label = label or split
        print("[Epoch {} {} SOTA]: MRR: {:.5}; H1: {:.5}; H3: {:.5}; H10: {:.5}".format(
            epoch, eval_label, results["mrr"], results["h1"], results["h3"], results["h10"]))
        return results

    def evaluate(self, split="valid", epoch=0, label=None):
        self.model.eval()

        queries = getattr(self.kg_data, f"{split}_triples")

        # Unified evaluation protocol:
        # 1) Bidirectional evaluation via inverse relation queries (tail pred + head pred via inverse)
        # 2) Filtered setting
        # 3) Tie-aware ranking
        # 4) Full-entity ranking (score against all entities)

        # Calculate offset for inverse relations: num_base_rel = num_rel / 2
        num_base_rel = self.args.num_rel // 2

        tail_queries = list(queries)
        head_queries = [(t, r + num_base_rel, h) for h, r, t in queries] if self.kg_data.add_inverse else []

        tail_results = self._run_eval_pass(tail_queries)["tie"]
        head_results = (self._run_eval_pass(head_queries)["tie"] if head_queries
                        else self._run_head_eval_pass(tail_queries)["tie"])

        results = {}
        for key in ('mrr', 'h1', 'h3', 'h10'):
            results[f'{key}_tail'] = tail_results[key]
            results[f'{key}_head'] = head_results[key]
            results[f'{key}_avg'] = (tail_results[key] + head_results[key]) / 2.0
        # Backwards-compatible aliases pointing at the bidirectional average
        results['mrr'] = results['mrr_avg']
        results['h1'] = results['h1_avg']
        results['h3'] = results['h3_avg']
        results['h10'] = results['h10_avg']

        eval_label = label or split
        print('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
            epoch, eval_label, results['mrr_tail'], results['mrr_head'], results['mrr_avg']))
        for k, key in ((1, 'h1'), (3, 'h3'), (10, 'h10')):
            print('[Epoch {} {}]: Hits@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
                epoch, eval_label, k, results[f'{key}_tail'], results[f'{key}_head'], results[f'{key}_avg']))

        return results

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
    from metrics_csv import upsert_metrics_csv, METRICS_CSV_HEADER

    def append_metrics_csv(output_path, dataset, model, metrics, seconds):
        upsert_metrics_csv(output_path, [
            dataset,
            model,
            f"{metrics['mrr_tail']:.5f}", f"{metrics['mrr_head']:.5f}", f"{metrics['mrr_avg']:.5f}",
            f"{metrics['h1_tail']:.5f}", f"{metrics['h1_head']:.5f}", f"{metrics['h1_avg']:.5f}",
            f"{metrics['h3_tail']:.5f}", f"{metrics['h3_head']:.5f}", f"{metrics['h3_avg']:.5f}",
            f"{metrics['h10_tail']:.5f}", f"{metrics['h10_head']:.5f}", f"{metrics['h10_avg']:.5f}",
            f"{seconds:.3f}",
        ])
    def append_sota_metrics_csv(output_path, dataset, model, metrics):
        upsert_metrics_csv(output_path, [
            dataset, model,
            f"{metrics['mrr']:.5f}", "—", "—",
            f"{metrics['h1']:.5f}", "—", "—",
            f"{metrics['h3']:.5f}", "—", "—",
            f"{metrics['h10']:.5f}", "—", "—", "—",
        ])

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../datasets/FB15K-237")
    parser.add_argument("--dataset", type=str, default="FB15K-237")
    parser.add_argument("--model", type=str, default="TransE")
    parser.add_argument("--gpu", type=int, default=0)
    
    # Defaults matching HoGRN 1-vs-All settings
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--tucker_rel_dim", type=int, default=0,
                        help="TuckER relation dim d_r (0 = same as emb_dim). "
                             "Smaller d_r shrinks the core tensor to fight overfitting on sparse data.")
    parser.add_argument("--ls_dackgr", type=int, choices=[0, 1], default=0,
                        help="Use DacKGR's label-smoothing additive (eps/N) instead of ConvE-standard 1/N.")
    parser.add_argument("--tucker_emb_drop", type=float, default=0.0,
                        help="TuckER embedding dropout on entity/relation/output-projection lookups "
                             "(DacKGR uses 0.3; this is the regularization the reimplementation omitted).")
    parser.add_argument("--margin", type=float, default=40.0, help="Gamma for TransE")
    parser.add_argument("--l2", type=float, default=0.0, help="Weight Decay")
    
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inp_drop", type=float, default=0.0, help="Input/embedding dropout (DistMult regularization)")
    parser.add_argument("--distmult_bn", type=int, default=0, help="Enable batch-norm on DistMult embeddings (0/1)")
    parser.add_argument("--add_inverse", type=int, choices=[0, 1], default=1,
                        help="Train reciprocal relations (1) or original triples only (0)")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "margin"],
                        help="Training regime: 'bce' (1-vs-All, HoGRN protocol) or "
                             "'margin' (classic pairwise margin-ranking, faithful TransE)")
    parser.add_argument("--margin_rank", type=float, default=1.0,
                        help="Ranking margin for --loss margin (classic TransE)")
    parser.add_argument("--selection_protocol", choices=["main", "sota"], default="main",
                        help="Validation metric used for checkpoint selection; evaluation outputs both")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load the existing checkpoint and only recompute both protocols")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Optional suffix for the checkpoint filename; set distinct "
                             "tags when running the same model+dataset concurrently")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_start = time.perf_counter()
    trainer = Trainer(args)
    if not args.eval_only:
        trainer.train()

    print("\nLoading best model for final evaluation...")
    ckpt_path = trainer.ckpt_path
    if os.path.exists(ckpt_path):
        trainer.model.load_state_dict(torch.load(ckpt_path, map_location=trainer.device))
        print("Evaluating on test set...")
        final_metrics = trainer.evaluate(split="test", epoch=0, label="test")
        sota_metrics = trainer.evaluate_sota(split="test", epoch=0, label="test")
        print(
            "SOTA_EVAL_METRICS baseline=traditional model={} dataset={} split=test "
            "mrr={:.5f} h1={:.5f} h3={:.5f} h10={:.5f}".format(
                args.model, args.dataset, sota_metrics["mrr"], sota_metrics["h1"],
                sota_metrics["h3"], sota_metrics["h10"]
            )
        )
        run_seconds = time.perf_counter() - run_start
        print(
            "FINAL_EVAL_METRICS baseline=traditional model={} dataset={} split=test "
            "mrr_tail={:.5f} mrr_head={:.5f} mrr_avg={:.5f} "
            "h1_tail={:.5f} h1_head={:.5f} h1_avg={:.5f} "
            "h3_tail={:.5f} h3_head={:.5f} h3_avg={:.5f} "
            "h10_tail={:.5f} h10_head={:.5f} h10_avg={:.5f}".format(
                args.model, args.dataset,
                final_metrics['mrr_tail'], final_metrics['mrr_head'], final_metrics['mrr_avg'],
                final_metrics['h1_tail'], final_metrics['h1_head'], final_metrics['h1_avg'],
                final_metrics['h3_tail'], final_metrics['h3_head'], final_metrics['h3_avg'],
                final_metrics['h10_tail'], final_metrics['h10_head'], final_metrics['h10_avg'],
            )
        )
        print("RUNTIME_STD baseline=traditional model={} dataset={} seconds={:.3f}".format(
            args.model, args.dataset, run_seconds))
        metrics_root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
        metrics_path = (
            os.path.join(metrics_root, "traditional_metrics.csv")
            if metrics_root
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "timings", "traditional_metrics.csv")
        )
        append_metrics_csv(
            metrics_path,
            args.dataset,
            args.model,
            final_metrics,
            run_seconds,
        )
        sota_metrics_path = (
            os.path.join(metrics_root, "traditional_sota_metrics.csv")
            if metrics_root
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "timings", "traditional_sota_metrics.csv")
        )
        append_sota_metrics_csv(sota_metrics_path, args.dataset, args.model, sota_metrics)
    else:
        print("No best model checkpoint found.")
