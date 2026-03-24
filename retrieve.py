from sentence_transformers import SentenceTransformer
import numpy as np
from index import BM25Index, HNSWIndex
from pydantic import BaseModel
from typing import Optional
import time
from utils import cal_ndcg


class RetrievedInfo(BaseModel):
    bm25_queries: list[str]
    hnsw_queries: list[str]
    bm25_objs: list[str]
    bm25_agg_pos_scores: list[float]
    bm25_agg_pos_scores_avg: list[float]
    bm25_agg_pos_scores_max: list[float]
    bm25_ids: list[list[int]]
    bm25_scores: list[list[float]]
    bm25_unique_ids: list[int]
    bm25_pos_scores: list[list[float]]
    hnsw_objs: list[str]
    hnsw_agg_pos_scores: list[float]
    hnsw_agg_pos_scores_avg: list[float]
    hnsw_agg_pos_scores_max: list[float]
    hnsw_ids: list[list[int]]
    hnsw_scores: list[list[float]]
    hnsw_unique_ids: list[int]
    hnsw_pos_scores: list[list[float]]


def retrieve_corpus(
    bm25_queries: list[str],
    hnsw_queries: list[str],
    corpus: list[str],
    args: dict,
    bm25_index: BM25Index,
    hnsw_index: HNSWIndex,
) -> RetrievedInfo:
    """
    Retrieve corpus from BM25 and HNSW index
    """
    start_time = time.time()
    bm25_ids, bm25_scores, unique_bm25_ids, pos_bm25_scores = bm25_index.search(
        bm25_queries, args.k
    )
    (
        bm25_agg_ids,
        bm25_agg_pos_scores,
        bm25_agg_pos_scores_avg,
        bm25_agg_pos_scores_max,
        pos_bm25_scores,
    ) = agg_results(
        unique_bm25_ids,
        pos_bm25_scores,
        args.k,
    )
    # print(f"Time for BM25 aggregation: {time.time() - start_time:.4f}s")
    bm25_objs = [corpus[i] for i in bm25_agg_ids]
    # print(f"Time for BM25 objects: {time.time() - start_time:.4f}s")

    hnsw_ids, hnsw_scores, query_embs, unique_hsnw_ids, results_embs = (
        hnsw_index.search(hnsw_queries, args.k)
    )
    # print(f"Time for HNSW retrieval: {time.time() - start_time:.4f}s")
    pos_hnsw_scores = np.dot(query_embs, results_embs.T)
    (
        hnsw_agg_ids,
        hnsw_agg_pos_scores,
        hnsw_agg_pos_scores_avg,
        hnsw_agg_pos_scores_max,
        pos_hnsw_scores,
    ) = agg_results(
        unique_hsnw_ids,
        pos_hnsw_scores,
        args.k,
    )
    # print(f"Time for HNSW aggregation: {time.time() - start_time:.4f}s")
    hnsw_objs = [corpus[i] for i in hnsw_agg_ids]
    retrieved_info = RetrievedInfo(
        bm25_queries=bm25_queries,
        hnsw_queries=hnsw_queries,
        bm25_objs=bm25_objs,
        bm25_agg_pos_scores=bm25_agg_pos_scores,
        bm25_agg_pos_scores_avg=bm25_agg_pos_scores_avg,
        bm25_agg_pos_scores_max=bm25_agg_pos_scores_max,
        bm25_ids=bm25_ids,
        bm25_scores=bm25_scores,
        bm25_unique_ids=unique_bm25_ids,
        bm25_pos_scores=pos_bm25_scores,
        hnsw_objs=hnsw_objs,
        hnsw_agg_pos_scores=hnsw_agg_pos_scores,
        hnsw_agg_pos_scores_avg=hnsw_agg_pos_scores_avg,
        hnsw_agg_pos_scores_max=hnsw_agg_pos_scores_max,
        hnsw_ids=hnsw_ids,
        hnsw_scores=hnsw_scores,
        hnsw_unique_ids=unique_hsnw_ids,
        hnsw_pos_scores=pos_hnsw_scores,
    )
    return retrieved_info


def combine_max_avg(pos_scores_avg, pos_scores_max, pos_ids, retrieved_ids):
    if len(pos_ids) > 0:
        best_beta = 0.5
        best_score = 0
        for i in range(0, 11):
            beta = i / 10
            scores = beta * pos_scores_avg + (1 - beta) * pos_scores_max
            sorted_ids, _ = zip(
                *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
            )
            ndcg_score = cal_ndcg(sorted_ids, pos_ids)
            if ndcg_score > best_score or (ndcg_score == best_score and beta == 0.5):
                best_score = ndcg_score
                best_beta = beta
        print(f"Best beta (beta * avg + (1 - beta) * max): {best_beta:.1f}")
        return best_beta * pos_scores_avg + (1 - best_beta) * pos_scores_max
    else:
        return 0.5 * pos_scores_avg + 0.5 * pos_scores_max


def choose_best_param(pos_scores, neg_scores, retrieved_ids, pos_ids) -> float:
    best_param = 0
    best_score = -1
    for param in range(0, 11):
        scores = pos_scores - neg_scores * param / 10
        sorted_ids, _ = zip(
            *sorted(zip(retrieved_ids, scores), key=lambda x: x[1], reverse=True)
        )
        ndcg_score = cal_ndcg(sorted_ids, pos_ids)
        if ndcg_score > best_score:
            best_score = ndcg_score
            best_param = param
    print(f"Best parameter: {best_param / 10:.1f}")
    return best_param / 10


def agg_results(
    unique_ids: list[int],
    pos_scores: list[list[float]],
    k: int,
) -> tuple[list[int], list[float], list[float], list[float], list[list[float]]]:
    """
    Aggregate BM25 results from multiple queries
    """
    pos_scores = np.array(pos_scores)
    max_scalar = max(np.max(pos_scores), 1e-6)
    print(f"max_scalar: {max_scalar:.4f}")
    if np.sum(pos_scores == max_scalar) > max(20, len(pos_scores)):
        max_scalar = max_scalar * 2
    pos_scores = pos_scores / max_scalar
    # min_scalar = np.min(pos_scores)
    pos_scores_avg = np.mean(pos_scores, axis=0)
    pos_scores_max = np.max(pos_scores, axis=0)
    agg_pos_scores = pos_scores_max
    retrieved_ids, agg_pos_scores, pos_scores_avg, pos_scores_max = zip(
        *sorted(
            zip(
                unique_ids,
                agg_pos_scores,
                pos_scores_avg,
                pos_scores_max,
            ),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    k = np.sum(np.array(pos_scores_max) > 0)
    print(f"retrieved num [k]: {k}")
    retrieved_ids = retrieved_ids[:k]
    agg_pos_scores = list(agg_pos_scores)[:k]
    pos_scores_avg = list(pos_scores_avg)[:k]
    pos_scores_max = list(pos_scores_max)[:k]
    return retrieved_ids, agg_pos_scores, pos_scores_avg, pos_scores_max, pos_scores
