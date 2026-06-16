import argparse
import math
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
import uuid
from data_utils import get_unique_entities, create_vocab, create_adj_map, load_data_all_triples, read_graph, get_inv_relation, load_data_by_vocab
from typing import *
from log import Logger
import json
import sys
from collections import deque

class StruPro(object):
    def __init__(self, args, train_map, eval_map, adj_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.adj_map = adj_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.num_non_executable_programs = []
        self.in_candidate = 0
        self.cached_neighbors = {}
        self.cached_programs = {}
        self.cached_num_ret_nn, self.cached_zero_ctr = {}, {}
        self.cached_rank_results = {}
        self.cached_siment = {}
        self.cached_collaboration, self.cached_success = {}, {}
        self.max_path_score, self.max_entity_score = 0, 0
        self.dist_matrix = {}

    # Cluster entities according to their types
    def cluster_entities_type(self, type_file):  #dict{e: type}, dict{type: set(e)}
        cluster_assignments_type, type2ents = {}, {}
        with open(type_file, "r") as fin:
            for _, line in enumerate(fin):
                e, t = line.strip().split("\t")
                cluster_assignments_type[e] = t
                if t not in type2ents:
                    type2ents[t] = set()
                type2ents[t].add(e)
        return cluster_assignments_type, type2ents
    
    def set_ansim(self, ansim):
        self.ansim = ansim

    def get_paths(self, e: int, ans: int):
        max_len = self.args.max_path_len
        all_programs = set()
        path = []
        visited = {e}
        def dfs(u: int):
            if u == ans:
                all_programs.add(tuple(path))
                return
            if len(path) >= max_len:
                return
            next_nodes = []
            for rel, v in self.adj_map.get(u, []):
                if v in visited:
                    continue
                if self.dist_matrix.get(v, {}).get(ans, float('inf')) > (max_len - len(path) - 1):
                    continue
                next_nodes.append((rel, v))
            if len(next_nodes) > self.args.max_path_branch:
                next_nodes = sorted(next_nodes, key=lambda x: self.dist_matrix[x[1]].get(ans, float('inf')))[:self.args.max_path_branch]
            for rel, v in next_nodes:
                path.append(rel)
                visited.add(v)
                dfs(v)
                visited.remove(v)
                path.pop()
        dfs(e)
        return list(all_programs)

    def rank_programs(self, list_programs: List[List[str]], r: int):
        if r in self.cached_rank_results:
            return self.cached_rank_results[r]
        nearest_entities = []
        # now get the score of each path
        path_and_scores = []
        for p in list_programs:
            trans_p = []
            for rel in p:
                trans_p.append(self.rel_vocab[rel])
            trans_p = tuple(trans_p)
            if len(trans_p) == 1 and trans_p[0] == r:
                continue
            try:
                score = self.args.precision_map[self.rev_rel_vocab[r]][p] * self.args.hop_factors[len(p)]
                path_and_scores.append((trans_p, score))
            except KeyError:
                # still a path or rel is missing.
                path_and_scores.append((trans_p, 0))

        # sort paths by their scores
        sorted_programs = sorted(path_and_scores, key=lambda item: -item[1])
        self.cached_rank_results[r] = sorted_programs
        for nn in range(self.args.train_ents):
            if len(self.train_map[nn, r]) > 0:
                nearest_entities.append(nn)

        # collaborations
        collaboration_map_filenm = os.path.join(self.args.data_dir, self.args.dataset, f"collaboration_map_{self.args.max_path_branch}branch.pkl")
        if os.path.exists(collaboration_map_filenm):
            return self.cached_rank_results[r]
        success_map, collaboration_map, self.cached_collaboration[r] = {}, {}, {}
        total_map, total_collaboration_map = {}, {}
        if len(sorted_programs) > 200:
            sorted_programs = sorted_programs[:200]
        for i in range(len(sorted_programs)):
            p1 = sorted_programs[i][0]
            collaboration_map[p1] = {}
            total_collaboration_map[p1] = {}
            self.cached_collaboration[r][p1] = {}
            success_map[p1] = 0
            total_map[p1] = 0
            for j in range(i + 1, len(sorted_programs)):
                p2 = sorted_programs[j][0]
                collaboration_map[p1][p2] = 0
                total_collaboration_map[p1][p2] = 0
        for e in nearest_entities:
            candidates = {}
            if len(self.train_map[(e, r)]) > 0:
                golden_answers = set(self.train_map[(e, r)])
                for p in sorted_programs:
                    p = p[0]
                    candidates[p] = set(self.execute_one_program(e, p).keys())
                    success_map[p] += len(candidates[p] & golden_answers)
                    total_map[p] += len(candidates[p])
                for i in range(len(sorted_programs)):
                    p1 = sorted_programs[i][0]
                    for j in range(i + 1, len(sorted_programs)):
                        p2 = sorted_programs[j][0]
                        collaborated_answers = candidates[p1] & candidates[p2]
                        collaboration_map[p1][p2] += len(collaborated_answers & golden_answers)
                        total_collaboration_map[p1][p2] += len(collaborated_answers)
        self.cached_success[r] = {}
        for p in success_map:
            if total_map[p] > 0:
                success_map[p] = success_map[p] / total_map[p]
                self.cached_success[r][p] = success_map[p]
        for p1 in collaboration_map:
            for p2 in collaboration_map[p1]:
                if total_collaboration_map[p1][p2] > 0:
                    collaborated_success = collaboration_map[p1][p2] / total_collaboration_map[p1][p2]
                    self.cached_collaboration[r][p1][p2] = collaborated_success
                    self.cached_collaboration[r][p2][p1] = collaborated_success
        return self.cached_rank_results[r]

    def execute_one_program(self, e: int, path: List[int]):
        cur_nodes, next_nodes = {e: 1}, {}
        for i in range(len(path)):
            if len(cur_nodes) == 0:
                break
            r = path[i]
            for e1 in cur_nodes:
                if (e1, r) in self.train_map:
                    for e2 in self.train_map[(e1, r)]:
                        if e2 not in next_nodes:
                            next_nodes[e2] = 0
                        next_nodes[e2] += cur_nodes[e1]
            cur_nodes = next_nodes
            next_nodes = {}
        return cur_nodes

    def execute_programs(self, e: str, r: str, path_list: list):
        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for i in range(len(path_list)):
            path, path_score = path_list[i]
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(e, path)
            temp = []
            path = tuple(path)
            for a in ans:
                temp.append((a, ans[a], path_score, path))
            ans = temp
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    def aggregate_answers(self, r, list_answers: List[Tuple[str, float, List[str], List[str]]]):
        score_map, score_sum = {}, {}
        ent_map = {}
        paths_to_e = {}
        likelihood_rates = []
        scale, diminishing_factor = 1.0, self.args.diminishing_factor

        # collect likelihood rates
        for i in range(len(list_answers)):
            e, e_times, e_score, path = list_answers[i]
            if e not in paths_to_e:
                paths_to_e[e] = set()
                score_sum[e] = 0
            path_score, cur_score = e_score, e_score
            for i in range(1, e_times):
                cur_score = cur_score * diminishing_factor
                path_score = path_score + cur_score
            score_sum[e] += path_score
            collab_sum, raw_sum = 0, 0
            for s, p in paths_to_e[e]:
                if r in self.cached_collaboration and p in self.cached_collaboration[r] and path in self.cached_collaboration[r][p]:
                    collab_sum += self.cached_collaboration[r][p][path]
                    s_original, e_score_original = s / self.args.hop_factors[len(p)], e_score / self.args.hop_factors[len(path)]
                    raw_sum += s_original + e_score_original - s_original * e_score_original
            if raw_sum > 0:
                likelihood_rate = collab_sum / raw_sum
                likelihood_rates.append(likelihood_rate)
            paths_to_e[e].add((e_score, path))
        
        # set likelihood rate range
        likelihood_range = (0.8, 1.2) # default range
        if len(likelihood_rates) > 0:
            mean_lr = np.mean(likelihood_rates)
            std_lr = np.std(likelihood_rates)
            likelihood_range = (mean_lr - std_lr, mean_lr + std_lr)
        
        # set scale rate
        if len(score_sum) > 0:
            scale = np.mean(list(score_sum.values()))
            scale = max(1.0, scale)

        for i in range(len(list_answers)):
            e, e_times, e_score, path = list_answers[i]
            if e not in score_map:
                score_map[e] = 0
                ent_map[e] = set()
                paths_to_e[e] = set()
            path_score = e_score / scale
            cur_score = path_score
            for i in range(1, e_times):
                cur_score = cur_score * diminishing_factor
                path_score = path_score + cur_score - path_score * cur_score
            collab_sum, raw_sum = 0, 0
            for s, p in paths_to_e[e]:
                if r in self.cached_collaboration and p in self.cached_collaboration[r] and path in self.cached_collaboration[r][p]:
                    collab_sum += self.cached_collaboration[r][p][path]
                    s_original, e_score_original = s / self.args.hop_factors[len(p)], e_score / self.args.hop_factors[len(path)]
                    raw_sum += s_original + e_score_original - s_original * e_score_original
            if raw_sum > 0:
                likelihood_rate = collab_sum / raw_sum
            else:
                likelihood_rate = likelihood_range[1]
            likelihood_rate = max(likelihood_range[0], min(likelihood_rate, likelihood_range[1]))
            if 0 < path_score < 1:
                odds = path_score / (1 - path_score)
                posterior_odds = odds * likelihood_rate
                path_score = posterior_odds / (1 + posterior_odds)
            score_map[e] = path_score + score_map[e] - path_score * score_map[e]
            ent_map[e] = ent_map[e] | self.cached_siment[r][path]
            paths_to_e[e].add((e_score, path))
            self.max_path_score = max(self.max_path_score, path_score)
            self.max_entity_score = max(self.max_entity_score, score_map[e])
            
        return score_map, ent_map
    
    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check[0]:
                return i + 1
        return -1

    def get_hits(self, answers, gold_answers, all_siment, query, dump_fh=None):
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        mr = 0.0
        (e1, r) = query
        e1_str = self.rev_entity_vocab[e1]
        r_str = self.rev_rel_vocab[r]
        all_gold_answers = self.args.all_kg_map[(e1_str, r_str)]
        all_gold_answers = [self.entity_vocab[x] for x in all_gold_answers]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = {}
            for pred in answers:
                if pred in all_gold_answers and pred != gold_answer:
                    continue
                else:
                    ss = 0
                    for ent in all_siment[pred]:
                        tmp = self.ansim[pred][ent]
                        ss = max(ss, tmp)
                    filtered_answers[pred] = answers[pred] * ss
            sorted_filtered_answers = sorted(filtered_answers.items(), key=lambda kv: -kv[1])

            # dump per-query info for external tie-aware bidirectional scorer
            if dump_fh is not None:
                gold_str = self.rev_entity_vocab[gold_answer]
                gold_sc = filtered_answers.get(gold_answer, 0.0)
                n_higher = sum(1 for s in filtered_answers.values() if s > gold_sc)
                n_tied = sum(1 for s in filtered_answers.values() if s == gold_sc)
                dump_fh.write(f"{e1_str}\t{r_str}\t{gold_str}\t{gold_sc:.8f}\t{n_higher}\t{n_tied}\n")

            rank = StruPro.get_rank_in_list(gold_answer, sorted_filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
                mr += rank
                self.in_candidate = self.in_candidate + 1
        return hits_10, hits_5, hits_3, hits_1, rr, mr

    def path_reasoning(self):
        num_programs = []
        num_answers = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr, mr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        dump_fh = None
        if getattr(self.args, 'dump_scores_file', None):
            dump_fh = open(self.args.dump_scores_file, 'w', buffering=1)
        for _, ((e1, r), e2_list) in enumerate(tqdm(self.eval_map.items())):
            # if e2_list is in train list then remove them
            # Normally, this shouldn't happen at all, but this happens for Nell-995.
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, self.rel_vocab, self.rev_rel_vocab, self.args.dataset)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            if r_inv is not None:
                for e2 in e2_list:
                    temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                    temp_list = []
                    for e1_dash in self.train_map[e2, r_inv]:
                        if e1_dash == e1:
                            continue
                        else:
                            temp_list.append(e1_dash)
                    self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            if e1 >= self.args.train_ents:
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                if r_inv is not None:
                    for e2 in e2_list:
                        self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue  # this entity was not seen during train; skip
 
            all_uniq_programs = list(self.args.precision_map[self.rev_rel_vocab[r]].keys())
            all_uniq_programs = self.rank_programs(all_uniq_programs, r)
            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, r, all_uniq_programs)
            answers, all_siment = self.aggregate_answers(r, answers)
            if len(answers) > 0:
                _10, _5, _3, _1, rr, mr1 = self.get_hits(answers, e2_list, all_siment, (e1, r), dump_fh=dump_fh)
                hits_10 += _10
                hits_5 += _5
                hits_3 += _3
                hits_1 += _1
                mrr += rr
                mr += mr1
                if self.args.output_per_relation_scores:
                    if r not in per_relation_scores:
                        per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0}
                        per_relation_query_count[r] = 0
                    per_relation_scores[r]["hits_1"] += _1
                    per_relation_scores[r]["hits_3"] += _3
                    per_relation_scores[r]["hits_5"] += _5
                    per_relation_scores[r]["hits_10"] += _10
                    per_relation_scores[r]["mrr"] += rr
                    per_relation_scores[r]["mr"] += mr1
                    per_relation_query_count[r] += len(e2_list)
            num_answers.append(len(answers))
            # put it back
            self.train_map[(e1, r)] = orig_train_e2_list
            if r_inv is not None:
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]

        if self.args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
                r_scores["mr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(self.args.output_dir, "per_relation_scores" + ".json")
            fout = open(out_file_name, "w")
            self.args.logger.info("Writing per-relation scores to {}".format(out_file_name))
            fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            fout.close()

        if dump_fh is not None:
            dump_fh.close()

        n = max(total_examples, 1)
        h1 = hits_1 / n
        h3 = hits_3 / n
        h5 = hits_5 / n
        h10 = hits_10 / n
        mrr_val = mrr / n

        self.args.logger.info(
            "Out of {} queries, at least one program was returned for {} queries".format(total_examples, non_zero_ctr))
        self.args.logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        self.args.logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        self.args.logger.info("Answer in candidate: {}, {}".format(self.in_candidate, self.in_candidate/total_examples))
        self.args.logger.info("{}/{}, Hits@1 {}".format(hits_1, total_examples, h1))
        self.args.logger.info("{}/{}, Hits@3 {}".format(hits_3, total_examples, h3))
        self.args.logger.info("{}/{}, Hits@5 {}".format(hits_5, total_examples, h5))
        self.args.logger.info("{}/{}, Hits@10 {}".format(hits_10, total_examples, h10))
        self.args.logger.info("{}/{}, MRR {}".format(mrr, total_examples, mrr_val))
        self.args.logger.info("{}/{}, MR {}".format(mr, total_examples, mr / total_examples))
        self.args.logger.info("Max path score: {}, Max entity score: {}".format(self.max_path_score, self.max_entity_score))
        self.args.logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(self.num_non_executable_programs)))

        # print structured line to stdout for the orchestrator to parse
        print(
            "STRUPROKGR_METRICS dataset={} split={} mrr={:.5f} h1={:.5f} h3={:.5f} h5={:.5f} h10={:.5f} n={}".format(
                self.args.dataset,
                "test" if self.args.test else "dev",
                mrr_val, h1, h3, h5, h10, total_examples,
            ),
            flush=True,
        )

        collaboration_map_filenm = os.path.join(self.args.data_dir, self.args.dataset, f"collaboration_map_{self.args.max_path_branch}branch.pkl")
        if not os.path.exists(collaboration_map_filenm):
            with open(collaboration_map_filenm, "wb") as fout:
                pickle.dump(self.cached_collaboration, fout)

    def distance_guided_path_collection(self, output_filenm=""):
        self.args.logger.info("Get all type-specific relation paths by distance-guided path collection")
        programs_map = {}
        sim_ent = {}
        for src in self.adj_map:
            dmap = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                du = dmap[u]
                if du >= self.args.max_path_len:
                    continue
                for rel, v in self.adj_map[u]:
                    if v not in dmap:
                        dmap[v] = du + 1
                        q.append(v)
            self.dist_matrix[src] = dmap
        for _, ((e1, r), e2_list) in enumerate(tqdm((self.train_map.items()))):
            r_id = r
            if r_id not in sim_ent:
                sim_ent[r_id] = {}
            r = self.rev_rel_vocab[r]
            c = self.args.cluster_assignments_type[self.rev_entity_vocab[e1]]
            if c not in programs_map:
                programs_map[c] = {}
            if r not in programs_map[c]:
                programs_map[c][r] = set()
            for nn_ans in e2_list:
                programs = self.get_paths(e1, nn_ans)
                for p in programs:
                    if len(p) == 1 and p[0] == r_id:
                        continue
                    if tuple(p) not in sim_ent[r_id]:
                        sim_ent[r_id][tuple(p)] = set()
                    sim_ent[r_id][tuple(p)].add(nn_ans)
                    p = [self.rev_rel_vocab[x] for x in p]
                    p = tuple(p)
                    programs_map[c][r].add(p)
        
        if not output_filenm:
            dir_name = os.path.join(self.args.data_dir, self.args.dataset)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            output_filenm = os.path.join(dir_name, f"type_relation_paths_{self.args.max_path_branch}branch.pkl")
        self.args.logger.info("Dumping relation-paths of entity types at {}".format(output_filenm))
        with open(output_filenm, "wb") as fout:
            pickle.dump(programs_map, fout)
        siment_filenm = os.path.join(self.args.data_dir, self.args.dataset, f"sim_ent_{self.args.max_path_branch}branch.pkl")
        self.args.logger.info("Dumping sim_ent at {}".format(siment_filenm))
        with open(siment_filenm, "wb") as fout:
            pickle.dump(sim_ent, fout)

    def calc_prec_supp_map(self, prec_supp_output_filenm=""):
        self.args.logger.info("Calculating precision and support map")
        success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
        train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        for ((e1, r), e2_list) in tqdm(train_map):
            r = self.rev_rel_vocab[r]
            c = self.args.cluster_assignments_type[self.rev_entity_vocab[e1]]
            if c not in success_map:
                success_map[c] = {}
            if c not in total_map:
                total_map[c] = {}
            if r not in success_map[c]:
                success_map[c][r] = {}
            if r not in total_map[c]:
                total_map[c][r] = {}
            for path in self.args.type_relation_paths[c][r]:
                trans_p = []
                for rel in path:
                    trans_p.append(self.rel_vocab[rel])
                trans_p = tuple(trans_p)
                ans = self.execute_one_program(e1, trans_p).keys()
                if len(ans) == 0:
                    continue
                # execute the path get answer
                if path not in success_map[c][r]:
                    success_map[c][r][path] = 0
                if path not in total_map[c][r]:
                    total_map[c][r][path] = 0
                for a in ans:
                    if a in e2_list:
                        success_map[c][r][path] += 1
                    total_map[c][r][path] += 1
        
        s_map_cross = {}
        total_map_cross = {}
        for c, _ in success_map.items():
            for r, _ in success_map[c].items():
                if r not in s_map_cross:
                    s_map_cross[r] = {}
                if r not in total_map_cross:
                    total_map_cross[r] = {}
                for path, s_c in success_map[c][r].items():
                    if path not in s_map_cross[r]:
                        s_map_cross[r][path] = 0
                    if path not in total_map_cross[r]:
                        total_map_cross[r][path] = 0
                    s_map_cross[r][path] = s_map_cross[r][path] + s_c
                    total_map_cross[r][path] = total_map_cross[r][path] + total_map[c][r][path]

        precision_map_cross = {}
        for r in s_map_cross:
            if r not in precision_map_cross:
                precision_map_cross[r] = {}
                for p in s_map_cross[r]:
                    precision_map_cross[r][p] = s_map_cross[r][p] / total_map_cross[r][p]
        self.args.logger.info("Dumping cross precision and support map at {}".format(prec_supp_output_filenm))
        prec_supp_map = {'prec': precision_map_cross, 'supp': total_map_cross}
        with open(prec_supp_output_filenm, "wb") as fout:
            pickle.dump(prec_supp_map, fout)
        self.args.logger.info("Dumping cross precision and support map: Done...")


def main(args):
    dataset = args.dataset
    args.logger.info("==========={}============".format(dataset))
    data_dir = os.path.join(args.data_dir, dataset)

    args.train_file = os.path.join(data_dir, "train.txt")
    args.dev_file = os.path.join(data_dir, "dev.txt")
    args.test_file = os.path.join(data_dir, "test.triples") if not args.test_file_name \
            else os.path.join(data_dir, args.test_file_name)
    args.logger.info("test_file:" + args.test_file)

    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, args.train_ents, args.train_rels = create_vocab(args.train_file, args.dev_file, args.test_file)
    args.logger.info("Loading train map")
    train_map = load_data_by_vocab(entity_vocab, rel_vocab, args.train_file)
    args.dev_file = os.path.join(data_dir, "dev.triples")
    args.logger.info("Loading dev map")
    dev_map = load_data_by_vocab(entity_vocab, rel_vocab, args.dev_file)
    args.logger.info("Loading test map")
    test_map = load_data_by_vocab(entity_vocab, rel_vocab, args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file
    adj_map = create_adj_map(args.train_file, entity_vocab, rel_vocab)

    # get the unique entities in eval set, so that we can calculate similarity in advance.
    eval_entities = get_unique_entities(eval_file)
    eval_vocab, eval_rev_vocab = {}, {}

    e_ctr = 0
    for e in eval_entities:
        eval_vocab[e] = e_ctr
        eval_rev_vocab[e_ctr] = e
        e_ctr += 1

    args.logger.info("=========Config:============")
    args.logger.info(vars(args))

    args.logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, args.dev_file, os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    StruPro_agent = StruPro(args, train_map, eval_map, adj_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab)

    # Calculate similarity
    adj_mat = read_graph(args.train_file, entity_vocab, rel_vocab)
    adj_mat = np.sqrt(adj_mat)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    l2norm = l2norm + np.finfo(float).eps
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)
    ansim = np.dot(adj_mat, adj_mat.T)
    StruPro_agent.set_ansim(ansim)
    
    # Cluster entities
    cluster_assignments_type, type2ents = StruPro_agent.cluster_entities_type(os.path.join(args.data_dir, args.dataset, args.type_file_name))  # dict{e: type}, dict{type: set(e)}
    args.cluster_assignments_type = cluster_assignments_type
    args.type2ents = type2ents

    # Get relation-paths of entity types
    type_relation_paths_filenm = os.path.join(data_dir, f"type_relation_paths_{args.max_path_branch}branch.pkl")
    if not os.path.exists(type_relation_paths_filenm):
        StruPro_agent.distance_guided_path_collection(output_filenm=type_relation_paths_filenm)
    args.logger.info("Loading relation-paths of entity types")
    with open(type_relation_paths_filenm, "rb") as fin:
        args.type_relation_paths = pickle.load(fin)
    siment_filenm = os.path.join(data_dir, f"sim_ent_{args.max_path_branch}branch.pkl")
    with open(siment_filenm, "rb") as fin:
        StruPro_agent.cached_siment = pickle.load(fin)

    # Get path scores
    prec_supp_map_filenm = os.path.join(data_dir, f"precision_support_map_{args.max_path_branch}branch.pkl")
    if not os.path.exists(prec_supp_map_filenm):
        StruPro_agent.calc_prec_supp_map(prec_supp_output_filenm=prec_supp_map_filenm)
    args.logger.info("Loading precision & support map")
    with open(prec_supp_map_filenm, "rb") as fin:
        prec_supp_map = pickle.load(fin)
        args.precision_map, args.support_map = prec_supp_map['prec'], prec_supp_map['supp']
    
    collaboration_map_filenm = os.path.join(data_dir, f"collaboration_map_{args.max_path_branch}branch.pkl")
    if os.path.exists(collaboration_map_filenm):
        args.logger.info("Loading collaboration map")
        with open(collaboration_map_filenm, "rb") as fin:
            StruPro_agent.cached_collaboration = pickle.load(fin)
    StruPro_agent.path_reasoning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="FB15K-237-10")
    parser.add_argument("--out_dir", type=str, default="outputs/")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='test.triples')
    parser.add_argument("--type_file_name", type=str, default="entity2type.txt")

    parser.add_argument("--max_num_programs", type=int, default=1000)
    parser.add_argument("--max_path_len", type=int, default=3)
    parser.add_argument("--max_path_branch", type=int, default=3)
    parser.add_argument("--decay_factor", type=float, default=0.95)
    parser.add_argument("--diminishing_factor", type=float, default=0.5)

    parser.add_argument("--name_of_run", type=str, default="unset")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    parser.add_argument("--dump_scores_file", type=str, default=None,
                        help="If set, write per-query (h, r, gold, score, n_higher, n_tied) TSV for external scoring")

    args = parser.parse_args()
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    args.logger = Logger(log_dir, args.name_of_run + "_" + str(args.max_path_branch), absolute=True).logger
    args.logger.info('\n\nCOMMAND: %s' % ' '.join(sys.argv))

    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]
    args.output_dir = os.path.join(args.out_dir, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logger.info(f"Output directory: {args.output_dir}")
    
    hop_factors = [0.0] * 11
    hop_factors[0], hop_factors[1] = 1.0, args.decay_factor
    for i in range(2, len(hop_factors)):
        hop_factors[i] = hop_factors[i-1] * args.decay_factor
    args.hop_factors = hop_factors
    main(args)
