import numpy as np
from constants import DIVERSITY_THRESHOLD
from scipy.stats import ttest_ind
from retrieve import RetrievedInfo
from collections import defaultdict
from index import BM25Index
from copy import deepcopy
import random
from utils import cal_ndcg, uct_selection


def leave_one_out(scores: list[float]) -> float:
    scores = sorted(scores, reverse=True)
    for s in scores:
        if s < 0.95:
            return s
    return 0.95


class Query:
    def __init__(self, org_query: str, reformat_template: str):
        self.org_query = org_query
        self.query_scores = {org_query: 2}  # store the results from rethinking
        self.bm25_query_scores = {org_query: 2}
        self.obj_scores = {}  # store the scores of the checked objects
        self.queries_from_generated = []
        self.queries_from_table = []
        self.new_queries_from_generated = []
        self.new_queries_from_table = []
        self.query_condition = reformat_template.format(query=org_query)
        self.reformulate_impact = 0
        self.query_list = [org_query]
        self.bm25_query_list = []
        self.pos_ids = []
        self.neg_ids = []
        self.last_query_list = []
        self.best_alpha = 0.5
        self.best_beta = 0.5
        self.id_to_obj = {}
        self.obj_to_id = {}
        self.obj_features = {}  # store the features of the checked objects
        self.pred_pos_objs = []
        self.non_linguistic_values = []
        self.last_obj_ids = []
        self.bm25_query_credits = defaultdict(int)
        self.hnsw_query_credits = defaultdict(int)
        self.bm25_query_visits = defaultdict(int)
        self.hnsw_query_visits = defaultdict(int)

    def score_retrieval(self, queries: list[str], ids: list[int], scores: dict):
        score_matrix = []
        for q in queries:
            score_matrix.append(scores[q])
        score_matrix = np.array(score_matrix)
        score = np.max(score_matrix, axis=0)  # [K]
        # rank the ids
        sorted_ids, _ = zip(*sorted(zip(ids, score), key=lambda x: x[1], reverse=True))
        # calculate ndcg with pos_ids
        ndcg_score = cal_ndcg(sorted_ids, self.pos_ids)
        return ndcg_score

    def select_query(
        self,
        queries: list[str],
        scores: dict,
        ids: list[int],
        non_linguistic_values=None,
    ):
        removed_queries, queries_to_check = [], deepcopy(queries)
        # iteratively remove the query with marginal score <= 0
        while len(queries_to_check) > 1:
            retrieval_score = self.score_retrieval(queries_to_check, ids, scores)
            # score each query
            query_scores = []
            for q in queries_to_check:
                queries_wo_q = [q1 for q1 in queries_to_check if q1 != q]
                retrieval_score_wo_q = self.score_retrieval(queries_wo_q, ids, scores)
                query_scores.append(retrieval_score - retrieval_score_wo_q)
            # get the smallest score
            min_score = min(query_scores)
            if min_score > 0:
                break
            # remove the query with the smallest score
            min_idx = query_scores.index(min_score)
            removed_queries.append(queries_to_check[min_idx])
            queries_to_check.pop(min_idx)
        print(f"Removed {len(removed_queries)} queries: {removed_queries}")
        # add new_queries from table
        queries_to_check.extend(self.new_queries_from_table)
        # queries not included in queries_to_check
        remained_queries = [
            q
            for q, s in self.query_scores.items()
            if s > 1 and q not in queries_to_check
        ]
        if non_linguistic_values is not None:
            remained_queries.extend(non_linguistic_values)
        # randomly select 10% queries from remained_queries
        sample_num = min(
            max(int(len(remained_queries) * 0.1), 2), len(remained_queries)
        )
        queries_to_check.extend(
            random.sample(remained_queries, sample_num) if sample_num > 0 else []
        )
        queries_to_check = list(set(queries_to_check))
        return queries_to_check

    def filter_ids_scores(self, ids: list[int], scores: list[float]):
        scores = np.array(scores)
        scores = scores.T  # [K, N]
        filtered_ids, filtered_scores = [], []
        for i, s in zip(ids, scores):
            if i in self.pos_ids or i in self.neg_ids:
                filtered_ids.append(i)
                filtered_scores.append(s)  # [K1, N]
        filtered_scores = np.array(filtered_scores).T  # [N, K1]
        return filtered_ids, filtered_scores

    def filter_sim_scores(self, unique_ids: list[int], scores: list[list[float]]):
        scores = np.array(scores)
        query_sim_thrs = []
        for i in range(scores.shape[1]):
            thr = np.sort(scores[:, i])[::-1][len(self.obj_scores) - 1]
            thr = max(thr, 1e-6)
            query_sim_thrs.append(thr)
        query_sim_scores = defaultdict(list)
        filtered_ids = []
        for id, score in zip(unique_ids, scores):
            if id in self.last_obj_ids:
                max_score = np.max(score)
                if max_score > 0:
                    filtered_ids.append(id)
                    for idx, s in enumerate(score):
                        query_sim_scores[idx].append(s)
        print(f"Filtered ids: {len(filtered_ids)}")
        return query_sim_thrs, query_sim_scores, filtered_ids

    def select_bm25_queries(
        self,
        retrieved_info: RetrievedInfo,
        select_query: str,
        non_linguistic_values: list,
    ):
        if select_query == "none" or retrieved_info is None:
            queries = [
                q for q, s in self.query_scores.items() if s > 1
            ] + non_linguistic_values
            queries = list(set(queries))
            return queries
        elif select_query == "random":
            candidates = [
                q for q, s in self.query_scores.items() if s > 1
            ] + non_linguistic_values
            candidates = list(set(candidates))
            return random.sample(candidates, min(len(candidates), 5))
        # evaluate the informativeness of each query and remove the non-informative queries
        bm25_queries = retrieved_info.bm25_queries  # N
        bm25_unique_ids = retrieved_info.bm25_unique_ids  # K
        bm25_pos_scores = retrieved_info.bm25_pos_scores  # [N, K]
        cnt = 0
        for i, q in enumerate(bm25_queries):
            if np.max(bm25_pos_scores[i]) <= 0:
                self.bm25_query_visits[q] = 1e6
                cnt += 1
            if q not in self.bm25_query_visits:
                self.bm25_query_visits[q] = 1e-6
        if cnt == len(bm25_queries):
            return [self.org_query]
        bm25_scores = np.array(bm25_pos_scores).T  # [K, N]
        query_sim_thrs, query_sim_scores, filtered_ids = self.filter_sim_scores(
            bm25_unique_ids, bm25_scores
        )
        for idx, q in enumerate(bm25_queries):
            sim = query_sim_thrs[idx]
            sim_scores = query_sim_scores[idx]
            credit, visit = 0, 0
            for s, obj_id in zip(sim_scores, filtered_ids):
                if s >= sim:
                    self.bm25_query_visits[q] += 1
                    visit += 1
                    if obj_id in self.pos_ids:
                        credit += 1
            new_credit = credit / visit if visit > 0 else 0
            if q in self.bm25_query_credits:
                self.bm25_query_credits[q] = (
                    self.bm25_query_credits[q] + new_credit
                ) / 2
            else:
                self.bm25_query_credits[q] = new_credit
        # get top queries with UCT selection
        queries = uct_selection(self.bm25_query_credits, self.bm25_query_visits)
        for q in self.new_queries_from_table:
            if q not in queries and self.bm25_query_visits.get(q, 0) < 1:
                queries.append(q)
        return queries

    def select_hnsw_queries(self, retrieved_info: RetrievedInfo, select_query: str):
        if select_query == "none" or retrieved_info is None:
            return [q for q, s in self.query_scores.items() if s > 1]
        elif select_query == "random":
            candidates = [q for q, s in self.query_scores.items() if s > 1]
            candidates = list(set(candidates))
            return random.sample(candidates, min(len(candidates), 5))
        # evaluate the informativeness of each query and remove the non-informative queries
        hnsw_queries = retrieved_info.hnsw_queries  # N
        hnsw_unique_ids = retrieved_info.hnsw_unique_ids  # K
        hnsw_pos_scores = retrieved_info.hnsw_pos_scores  # [N, K]
        hnsw_scores = np.array(hnsw_pos_scores).T  # [K, N]
        for q in hnsw_queries:
            if q not in self.hnsw_query_visits:
                self.hnsw_query_visits[q] = 1e-6
        query_sim_thrs, query_sim_scores, filtered_ids = self.filter_sim_scores(
            hnsw_unique_ids, hnsw_scores
        )
        sim_matrix = []
        for idx, q in enumerate(hnsw_queries):
            sim_matrix.append(query_sim_scores[idx])  # [N, K1]
        sim_matrix = np.array(sim_matrix)
        for idx, q in enumerate(hnsw_queries):
            sim = query_sim_thrs[idx]
            sim_scores = query_sim_scores[idx]
            credit, visit = 0, 0
            for j, (s, obj_id) in enumerate(zip(sim_scores, filtered_ids)):
                if s >= sim:
                    self.hnsw_query_visits[q] += 1
                    visit += 1
                    if obj_id in self.pos_ids and s == np.max(sim_matrix[:, j]):
                        credit += 1
            new_credit = credit / visit if visit > 0 else 0
            if q in self.hnsw_query_credits:
                self.hnsw_query_credits[q] = (
                    self.hnsw_query_credits[q] + new_credit
                ) / 2
            else:
                self.hnsw_query_credits[q] = new_credit
        # get top queries with UCT selection
        queries = uct_selection(self.hnsw_query_credits, self.hnsw_query_visits)
        for q in self.new_queries_from_table:
            if q not in queries and self.hnsw_query_visits.get(q, 0) < 1:
                queries.append(q)
        return queries

    def select_bm25_query_words(self, select_query):
        if select_query == "none":
            return list(self.query_scores.keys())
        query_list = []
        query_words_list = []
        # keep the query words with score > 1
        if (
            np.sum(np.array(list(self.query_scores.values())) > 0) == 1
            or select_query == "diversified"
        ):
            thr = 0
        else:
            thr = 2
        for q, s in self.query_scores.items():
            if s >= thr:
                words = q.split()
                flag = True
                if select_query == "diversified":
                    for selected_q_word in query_words_list:
                        if len(set(selected_q_word) - set(words)) == 0:
                            flag = False
                            break
                if flag:
                    query_list.append(q)
                    query_words_list.append(words)

        if len(self.obj_scores) == 0 or sum(list(self.obj_scores.values())) == 0:
            self.bm25_query_list = query_list
            return query_list
        # summarize the word frequency in positive and negative samples
        pos_frequency = defaultdict(int)
        neg_frequency = defaultdict(int)
        for obj, score in self.obj_scores.items():
            if score > 0:
                for word in obj.split():
                    pos_frequency[word.lower()] += 1
            else:
                for word in obj.split():
                    neg_frequency[word.lower()] += 1
        new_query_list = []
        for query in query_list:
            words = query.split()
            word_ratio = {}
            ratio_thr = 1
            for word in words:
                pos_freq = pos_frequency.get(word.lower(), 0)
                neg_freq = neg_frequency.get(word.lower(), 0)
                word_ratio[word.lower()] = pos_freq / (pos_freq + neg_freq + 1e-6)
                if pos_freq > 0 and pos_freq / (pos_freq + neg_freq + 1e-6) < ratio_thr:
                    ratio_thr = pos_freq / (pos_freq + neg_freq + 1e-6)
            for word, ratio in word_ratio.items():
                repeated_num = min(int(ratio / ratio_thr) - 1, 5)
                if repeated_num > 0:
                    words.extend([word] * repeated_num)
                    # words.append(word)
            new_query_list.append(" ".join(words))
        self.bm25_query_list = new_query_list
        return new_query_list

    def select_diversified_query_words(self, emb_model, select_query):
        if select_query == "none":
            return list(self.query_scores.keys())
        query_list = []
        # keep the query words with score > 1
        if (
            np.sum(np.array(list(self.query_scores.values())) > 0) == 1
            or select_query == "diversified"
        ):
            thr = 0
        else:
            thr = 2
        for q, s in self.query_scores.items():
            if s >= thr:
                query_list.append(q)
        if select_query == "reliable":
            return query_list
        selected_query_list = []
        selected_qids = []
        for i, q in enumerate(query_list):
            if q == self.org_query:
                selected_qids.append(i)
                selected_query_list.append(q)
        if len(query_list) == len(selected_query_list):
            self.query_list = selected_query_list
            return selected_query_list
        pos_embs = emb_model.encode(query_list)
        # normalize the pos embs
        pos_embs = pos_embs / np.linalg.norm(pos_embs, axis=1, keepdims=True)
        sim_matrix = np.dot(pos_embs, pos_embs.T)
        threshold = np.median(sim_matrix)
        print(f"Threshold: {threshold}")
        sim_dist = np.round(np.percentile(sim_matrix, list(range(0, 100, 10))), 4)
        print(f"Distribution of similarity scores: {list(sim_dist)}")
        # select the diversified query words from the query list
        sim_scores = np.max(sim_matrix[selected_qids], axis=0)
        # prioritize the newly generated/verified query words
        for i, q in enumerate(query_list):
            if (
                q not in self.new_queries_from_generated
                and q not in self.new_queries_from_table
            ):
                sim_scores[i] = 1
        for i in range(len(query_list) - len(selected_query_list)):
            # select the word with the least similarity score
            qid = np.argmin(sim_scores)
            if sim_scores[qid] > max(DIVERSITY_THRESHOLD, threshold):
                break
            sim_scores = np.maximum(sim_scores, sim_matrix[qid])
            selected_query_list.append(query_list[qid])
            selected_qids.append(qid)
        sim_scores = np.max(sim_matrix[selected_qids], axis=0)
        # select the remaining query words
        for i in range(len(query_list) - len(selected_query_list)):
            # select the word with the least similarity score
            qid = np.argmin(sim_scores)
            if sim_scores[qid] > max(DIVERSITY_THRESHOLD, threshold):
                break
            sim_scores = np.maximum(sim_scores, sim_matrix[qid])
            selected_query_list.append(query_list[qid])
            selected_qids.append(qid)
        self.query_list = selected_query_list
        print(f"Select {len(selected_query_list)} from {len(query_list)}")
        return selected_query_list

    def update_queries_from_generated(self, new_queries_from_generated: list):
        self.new_queries_from_generated = new_queries_from_generated
        self.queries_from_generated.extend(new_queries_from_generated)

    def update_queries_from_table(self, new_queries_from_table: list):
        self.new_queries_from_table = new_queries_from_table
        self.queries_from_table.extend(new_queries_from_table)

    def update_query_scores(self, query_scores: dict):
        self.query_scores.update(query_scores)

    def update_bm25_query_scores(self, bm25_query_scores: dict):
        self.bm25_query_scores.update(bm25_query_scores)

    def hnsw_leave_one_out(self, scores: list[float], obj: str) -> float:
        if (
            obj in self.query_list
            and obj not in self.queries_from_generated
            and obj != self.org_query
        ):
            idx = self.query_list.index(obj)
            scores[idx] = 0
        return np.max(scores)

    def bm25_leave_one_out(self, scores: list[float], obj: str) -> float:
        if (
            obj in self.bm25_query_list
            and obj not in self.queries_from_generated
            and obj != self.org_query
        ):
            idx = self.bm25_query_list.index(obj)
            scores[idx] = 0
        return np.max(scores)

    def update_obj_features(self, retrieved_info: RetrievedInfo):
        bm25_unique_ids = retrieved_info.bm25_unique_ids
        hnsw_unique_ids = retrieved_info.hnsw_unique_ids
        bm25_pos_scores = retrieved_info.bm25_pos_scores
        hnsw_pos_scores = retrieved_info.hnsw_pos_scores

        for obj in self.obj_scores:
            if obj in self.obj_features:
                continue
            obj_id = self.obj_to_id[obj]
            try:
                bm25_index = bm25_unique_ids.index(obj_id)
                bm25_pos_score = [v[bm25_index] for v in bm25_pos_scores]
                bm25_max_score = self.bm25_leave_one_out(bm25_pos_score, obj)
                bm25_avg_score = np.mean(bm25_pos_score)
            except:
                bm25_max_score = 0
                bm25_avg_score = 0
            try:
                hnsw_index = hnsw_unique_ids.index(obj_id)
                hnsw_pos_score = [v[hnsw_index] for v in hnsw_pos_scores]
                hnsw_max_score = self.hnsw_leave_one_out(hnsw_pos_score, obj)
                hnsw_avg_score = np.mean(hnsw_pos_score)
            except:
                hnsw_max_score = 0
                hnsw_avg_score = 0
            self.obj_features[obj] = [
                bm25_max_score,
                bm25_avg_score,
                hnsw_max_score,
                hnsw_avg_score,
            ]

    def update_obj_scores(self, obj_scores: dict, corpus: list):
        self.obj_scores.update(obj_scores)
        pos_ids, neg_ids = [], []
        for obj, score in obj_scores.items():
            if score > 0:
                try:
                    pos_ids.append(self.obj_to_id[obj])
                except:
                    pos_id = corpus.index(obj)
                    self.obj_to_id[obj] = pos_id
                    self.id_to_obj[pos_id] = obj
                    pos_ids.append(pos_id)
                if self.query_scores.get(obj, 0) == 2 and obj not in self.pred_pos_objs:
                    self.pred_pos_objs.append(obj)
            else:
                try:
                    neg_ids.append(self.obj_to_id[obj])
                except:
                    neg_id = corpus.index(obj)
                    self.obj_to_id[obj] = neg_id
                    self.id_to_obj[neg_id] = obj
                    neg_ids.append(neg_id)
        self.pos_ids.extend(pos_ids)
        self.neg_ids.extend(neg_ids)
        self.last_obj_ids = pos_ids + neg_ids
