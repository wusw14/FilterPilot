from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import hnswlib
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import time

# nltk.download("stopwords")
# nltk.download("punkt")


class Index:
    def __init__(self, corpus: list[str], dataset_name: str):
        self.corpus = corpus
        self.dataset_name = dataset_name

    def build_index(self):
        pass

    def search(self, query: list[str], top_n: int = 1000):
        pass

    def agg_results(self, results: list[list[str]], scores: list[list[float]]):
        pass


class BM25Index(Index):
    def __init__(self, corpus: list[str], dataset_name: str):
        super().__init__(corpus, dataset_name)
        self.index_path = f"index/{dataset_name}_bm25_index.pkl"
        self.stop_words = set(stopwords.words("english"))
        self.index = self.build_index()

    def build_index(self) -> BM25Okapi:
        """
        Build BM25 index
        Load index from file if exists, otherwise build index and save to file
        """
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        # tokenization
        tokenized_corpus = []
        for doc in self.corpus:
            doc = doc.replace("/", " ")
            doc = doc.replace("-", " ")
            tokens = word_tokenize(doc.lower())
            tokens = [token for token in tokens if token not in self.stop_words]
            tokenized_corpus.append(tokens)
        # build BM25 index
        bm25 = BM25Okapi(tokenized_corpus, b=0.1)
        # save index
        with open(self.index_path, "wb") as f:
            pickle.dump(bm25, f)
        return bm25

    def search(
        self, query: list[str], top_n: int = 1000
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
        """
        Search BM25 index
        Return results, scores, full_scores
        """
        results = []
        scores = []
        full_scores = []
        unique_results = set()
        for q in query:
            # tokenization
            q = q.replace("/", " ")
            q = q.replace("-", " ")
            tokenized_query = [
                token
                for token in word_tokenize(q.lower())
                if token not in self.stop_words
            ]
            score = self.index.get_scores(tokenized_query)
            # norm_term = np.sum([self.index.idf.get(q) or 0 for q in tokenized_query])
            score = score / len(tokenized_query)
            full_scores.append(score)
            result = np.argsort(score)[::-1][:top_n].tolist()  # get top_n indices
            score = np.array(score)[result]
            results.append(result)
            scores.append(score)
            unique_results.update(result)
        unique_results = list(unique_results)
        unique_scores = np.array(full_scores)[:, unique_results]
        return results, scores, unique_results, unique_scores

    def filter_query_objs(self, query_objs: list[str]) -> list[str]:
        filtered_query_objs = []
        for q in query_objs:
            q = q.replace("/", " ")
            q = q.replace("-", " ")
            tokenized_query = [
                token
                for token in word_tokenize(q.lower())
                if token not in self.stop_words
            ]
            obj_tokens = []
            for token in tokenized_query:
                idf = self.index.idf.get(token, 0)
                if idf > 0:
                    obj_tokens.append(token)
            if len(obj_tokens) > 0:
                filtered_query_objs.append(" ".join(obj_tokens))
        return list(set(filtered_query_objs))

    def get_neg_scores(
        self, query: list[str], unique_results: list[int]
    ) -> list[list[float]]:
        if len(query) == 0:
            return None
        scores = []
        for q in query:
            # tokenization
            tokenized_query = [
                token
                for token in word_tokenize(q.lower())
                if token not in self.stop_words
            ]
            score = self.index.get_batch_scores(tokenized_query, unique_results)
            scores.append(score)
        return scores


class HNSWIndex:
    def __init__(self, corpus: list[str], dataset_name: str):
        self.corpus = corpus
        self.dataset_name = dataset_name
        self.emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
        self.dim = 768
        self.sim_metric = "cosine"
        self.index_path = f"index/{dataset_name}_hnsw_index_{self.dim}.bin"
        self.index = self.build_index()

    def build_index(self) -> hnswlib.Index:
        """
        Build HNSW index
        Load index from file if exists, otherwise build index and save to file
        """
        if os.path.exists(self.index_path):
            # load HNSW index
            p = hnswlib.Index(space=self.sim_metric, dim=self.dim)
            p.load_index(self.index_path)
            return p

        embs = self.emb_model.encode(self.corpus, batch_size=512)
        num_elements = embs.shape[0]
        dim = embs.shape[1]
        print(f"num elements: {num_elements}, dim: {dim}")

        # Declaring index
        p = hnswlib.Index(
            space="cosine", dim=self.dim
        )  # possible options are l2, cosine or ip
        p.init_index(max_elements=num_elements, ef_construction=200, M=40)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(220)

        # Set number of threads used during batch search/construction
        # By default using all available cores
        # p.set_num_threads(4)
        p.add_items(embs)

        # Serializing and deleting the index:
        print("Saving index to '%s'" % self.index_path)
        p.save_index(self.index_path)
        return p

    def search(
        self, query: list[str], top_n: int = 1000
    ) -> tuple[list[list[int]], list[list[float]], np.ndarray, list[int], np.ndarray]:
        """
        Search HNSW index
        Return results, scores, query_embedding, unique_results, unique_results_embs
        """
        # encode query
        start_time = time.time()
        query = [q.lower() for q in query]
        query_embedding = self.emb_model.encode(query, batch_size=512)
        # print(f"Time for HNSW query encoding: {time.time() - start_time:.4f}s")
        # search for nearest neighbors
        labels, distances = self.index.knn_query(query_embedding, k=top_n)
        # print(f"Time for HNSW search: {time.time() - start_time:.4f}s")
        results = []
        scores = []
        unique_results = []
        for i, q in enumerate(query):
            result, score = [], []
            for j in range(top_n):
                doc_id = int(labels[i][j])
                result.append(doc_id)
                score.append(1 - distances[i][j])
            results.append(result)
            scores.append(score)
            unique_results.extend(result)
        unique_results = list(set(unique_results))
        # get the embeddings of the unique results
        unique_results_embs = self.index.get_items(unique_results, return_type="numpy")
        # print(f"Time for HNSW unique results: {time.time() - start_time:.4f}s")
        return results, scores, query_embedding, unique_results, unique_results_embs
