import time
import uuid
import threading
import os
import sqlite3
import hashlib
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from load_data import load_data
from index import BM25Index, HNSWIndex
from reformulate import reformulate, reformulate_simple
from retrieve import retrieve_corpus
from rerank import rerank_retrieved_objs
from iterative_check import llm_check_retrieved_objs
from query import Query
from constants import CHECK_NUM
from pathlib import Path
from collections import Counter

DB_ROOT = Path("./dataset").resolve()
SUPPORTED_DATASETS = ["animal", "chemical_compound", "product"]


# =========================================================
# API Models
# =========================================================
class StartReq(BaseModel):
    db_path: str
    table: str
    column: Optional[str] = None
    value: str = Field(..., min_length=1)
    timeLimitSec: Optional[int] = Field(100, ge=1, le=600)


class StartResp(BaseModel):
    sessionId: str
    reformulatedTerms: List[str]
    t1Sec: float


class IterateReq(BaseModel):
    sessionId: str
    iteration: int = Field(..., ge=1)


class IterateResp(BaseModel):
    t1Sec: float
    reformulatedTerms: List[str]
    alignedHigh: List[str]
    alignedPotential: List[str]
    hnswQueries: List[str]
    bm25Queries: List[str]
    summary: str
    done: bool
    suggestedStop: bool
    timeUp: bool
    checkedTotal: int  # len(query.obj_scores), cumulative LLM-checked objects


# =========================================================
# Runtime Args
# =========================================================
class ArgsLite:
    def __init__(self):
        self.dataset = "product"

        # LLM-based reformulation：只在 start 做一次
        self.reform_type = "multi-aspect"

        self.budget = 500
        self.iterative_check = True

        # query selection
        self.select_query = "uct"

        self.index_combine_method = "weighted"
        self.llm_template = (
            "Is '{value}' the same as or a type of '{query}'? "
            "Directly answer with 'Yes', 'No', or 'Unsure'."
        )

        self.alpha = 1.0
        self.tau = 0.2
        self.k = 1000

        # rerank / LLM check
        self.top_k = 20
        self.steps = 5
        self.early_stop = True
        self.rerank = "max"


# =========================================================
# Session State
# =========================================================
class SessionState:
    """state of a FilterPilot session（supports soft stop）"""

    def __init__(
        self,
        session_id: str,
        args: ArgsLite,
        attribute: str,
        corpus: List[str],
        bm25_index: BM25Index,
        hnsw_index: HNSWIndex,
        reformat_template: str,
        org_query: str,
        ttl_sec: int,
    ):
        self.session_id = session_id
        self.args = args
        self.attribute = attribute
        self.corpus = corpus
        self.bm25_index = bm25_index
        self.hnsw_index = hnsw_index

        self.query = Query(org_query, reformat_template)

        self.iteration = 0
        self.retrieved_info = None
        self.non_linguistic_values: List[str] = []

        # ---- stopping related states ----
        self.done = False  # hard stop（budget）
        self.recommended_stop = False  # soft stop（IQE+）
        self.time_up = False  # soft stop（time limit）
        self.early_stop = 0  # 连续无进展计数（per-session）

        self.expires_at = time.time() + ttl_sec
        print(f"[DEBUG] expires_at={self.expires_at:.2f}, ttl_sec={ttl_sec:.2f}")


class PathCheckReq(BaseModel):
    # 兼容不同前端字段命名：既支持 `path` 也支持 `db_path`
    path: Optional[str] = None
    db_path: Optional[str] = None

    def get_db_path(self) -> Path:
        raw = self.db_path if self.db_path else self.path
        if not raw:
            raise ValueError("Missing db_path/path")
        return Path(raw).expanduser().resolve()


def resolve_dataset_root(db_path: str | Path) -> tuple[Path, Path]:
    """
    统一 `db_path` 的语义：
    - 允许用户传入包含 `dataset/` 的上级目录
    - 或者直接传入 `dataset/` 目录本身
    返回值：
    - dataset_root: 实际的 dataset 根目录（里面有 animal/chemical_compound/product）
    - project_root: 用于 os.chdir 的目录（使得 `dataset/{name}/...` 相对可用）
    """
    p = Path(db_path).expanduser().resolve()

    # 情况 A：用户传上级目录（里面有 dataset/）
    if (p / "dataset").exists() and (p / "dataset").is_dir():
        dataset_root = (p / "dataset").resolve()
        project_root = p
        return dataset_root, project_root

    # 情况 B：用户直接传 dataset/ 目录
    dataset_root = p
    project_root = p.parent
    return dataset_root, project_root


def list_available_datasets(dataset_root: Path) -> list[str]:
    available: list[str] = []
    for name in SUPPORTED_DATASETS:
        csv_path = dataset_root / name / f"{name}.csv"
        qa_path = dataset_root / name / "query_answer.json"
        if csv_path.exists() and qa_path.exists():
            available.append(name)
    return available


# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(title="IQE API", version="paper-soft-stop")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GLOBAL: Dict[str, dict] = {}
SESSIONS: Dict[str, SessionState] = {}

GLOBAL_LOCK = threading.Lock()
SESS_LOCK = threading.Lock()


# =========================================================
# Utilities
# =========================================================
def ensure_dataset_loaded(
    dataset: str, column_hint: Optional[str], base_dir: Path | None = None
) -> dict:
    if base_dir is None:
        base_dir = Path.cwd()
    cache_key = (dataset, str(base_dir.resolve()))

    with GLOBAL_LOCK:
        if cache_key in GLOBAL:
            return GLOBAL[cache_key]

        df, *_ = load_data(dataset)

        attribute = "Product_Title" if dataset == "product" else df.columns[0]
        if column_hint and column_hint in df.columns:
            attribute = column_hint

        corpus = df[attribute].dropna().astype(str).tolist()

        bm25_index = BM25Index(corpus, dataset)
        hnsw_index = HNSWIndex(corpus, dataset)

        GLOBAL[cache_key] = {
            "df": df,
            "attribute": attribute,
            "corpus": corpus,
            "bm25": bm25_index,
            "hnsw": hnsw_index,
        }

        return GLOBAL[cache_key]


def _quote_ident(name: str) -> str:
    """
    Quote SQLite identifiers safely with double-quotes.

    Note: this is intended only for trusted schema-derived identifiers
    (table/column names read from sqlite_master/PRAGMA).
    """

    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def _sqlite_schema(db_path: Path) -> dict:
    """
    Returns schema in the same shape as frontend expects:
    { "datasets": [ { "name": tableName, "columns": [col1, col2, ...] }, ... ] }
    """

    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"db file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [r["name"] for r in cur.fetchall()]

        datasets: list[dict] = []
        for t in tables:
            cur.execute(f"PRAGMA table_info({_quote_ident(t)})")
            cols = [row["name"] for row in cur.fetchall() if row["name"]]
            if cols:
                datasets.append({"name": t, "columns": cols})

        return {"datasets": datasets}
    finally:
        conn.close()


def _sqlite_make_index_name(db_path: Path, table: str, column: str) -> str:
    # Keep index filenames short/stable.
    digest = hashlib.sha1(f"{db_path}|{table}|{column}".encode("utf-8")).hexdigest()[
        :12
    ]
    return f"sqlite_{digest}"


def _sqlite_load_column_values(db_path: Path, table: str, column: str) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        sql = (
            f"SELECT {_quote_ident(column)} FROM {_quote_ident(table)} "
            f"WHERE {_quote_ident(column)} IS NOT NULL"
        )
        cur.execute(sql)
        values: list[str] = []
        for row in cur.fetchall():
            v = row[0]
            if v is None:
                continue
            sv = str(v).strip()
            if sv:
                values.append(sv)
        return values
    finally:
        conn.close()


def ensure_sqlite_dataset_loaded(
    db_path: Path, table: str, column_hint: Optional[str]
) -> dict:
    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"db file not found: {db_path}")

    schema = _sqlite_schema(db_path)
    datasets = schema.get("datasets") or []
    found = next((d for d in datasets if d["name"] == table), None)
    if not found:
        raise ValueError(f"Invalid table: {table}")

    columns: list[str] = found.get("columns") or []
    if not columns:
        raise ValueError(f"Table has no columns: {table}")

    attribute = column_hint if (column_hint and column_hint in columns) else columns[0]

    cache_key = ("sqlite", str(db_path.resolve()), table, attribute)
    with GLOBAL_LOCK:
        if cache_key in GLOBAL:
            return GLOBAL[cache_key]

    # Ensure index directory exists before BM25/HNSW try to write files.
    Path("index").mkdir(exist_ok=True)

    corpus = _sqlite_load_column_values(db_path, table, attribute)
    # select the most frequent values as samples
    samples = [v[0] for v in Counter(corpus).most_common(5)]
    corpus = list(set(corpus))
    if not corpus:
        raise ValueError(f"No values found in {table}.{attribute}")

    index_name = _sqlite_make_index_name(db_path, table, attribute)
    bm25_index = BM25Index(corpus, index_name)
    hnsw_index = HNSWIndex(corpus, index_name)
    samples = corpus[:5]

    with GLOBAL_LOCK:
        GLOBAL[cache_key] = {
            "df": None,
            "attribute": attribute,
            "corpus": corpus,
            "bm25": bm25_index,
            "hnsw": hnsw_index,
            "samples": samples,
        }
        return GLOBAL[cache_key]


def do_reformulate(
    args: ArgsLite,
    org_query: str,
    attribute: str,
    samples: List[str],
) -> List[str]:
    if args.reform_type == "multi-aspect":
        func = reformulate
    elif args.reform_type == "simple":
        func = reformulate_simple
    else:
        return []

    gen_terms, _ = func(org_query, attribute, samples, None)
    return [q for q in gen_terms if q != org_query]


def _get_session_or_404(sid: str) -> SessionState:
    with SESS_LOCK:
        st = SESSIONS.get(sid)
        if not st:
            raise HTTPException(status_code=404, detail="Invalid sessionId")

        # soft time limit
        print(f"[DEBUG] current time={time.time():.2f}, expires_at={st.expires_at:.2f}")
        if time.time() > st.expires_at:
            st.time_up = True

        return st


# =========================================================
# META API
# =========================================================
@app.get("/api/meta")
def meta():
    datasets = []
    for name in ["animal", "chemical_compound", "product"]:
        try:
            df, *_ = load_data(name)
            datasets.append({"name": name, "columns": list(df.columns)})
        except Exception as e:
            print(f"[META] Failed to load {name}: {e}")
    return {"datasets": datasets}


@app.post("/api/check_db_path")
def check_db_path(req: PathCheckReq):
    try:
        p = req.get_db_path()
    except Exception:
        return {"ok": False, "message": "Invalid db_path"}

    # SQLite mode: allow passing a db file directly.
    if p.exists() and p.is_file():
        try:
            schema = _sqlite_schema(p)
        except Exception as e:
            return {"ok": False, "message": f"Invalid sqlite db: {e}"}
        datasets = schema.get("datasets") or []
        if not datasets:
            return {"ok": False, "message": "sqlite db has no user tables."}
        return {"ok": True, "message": "db file is valid.", "datasets": datasets}

    # Folder mode (legacy): allow passing a folder (or its parent).
    dataset_root, _project_root = resolve_dataset_root(str(p))
    if not dataset_root.exists() or not dataset_root.is_dir():
        return {
            "ok": False,
            "message": "Invalid db_path: dataset root not found.",
        }

    available = list_available_datasets(dataset_root)
    if not available:
        return {
            "ok": False,
            "message": "No supported dataset found under provided `db_path`.",
        }

    return {"ok": True, "message": "db_path is valid.", "datasets": available}


import os


class MetaForPathReq(BaseModel):
    db_path: str


@app.post("/api/meta_for_path")
def meta_for_path(req: MetaForPathReq):
    try:
        p = Path(req.db_path).expanduser().resolve()
    except Exception:
        raise HTTPException(400, "Invalid db_path")

    # SQLite mode: db_path is a sqlite file.
    if p.exists() and p.is_file():
        try:
            schema = _sqlite_schema(p)
        except Exception as e:
            raise HTTPException(400, f"Failed to read sqlite schema: {e}")
        datasets = schema.get("datasets") or []
        if not datasets:
            raise HTTPException(400, "sqlite db has no user tables")
        return {"datasets": datasets}

    # Folder mode: legacy CSV-based demo datasets.
    dataset_root, project_root = resolve_dataset_root(str(p))
    available = list_available_datasets(dataset_root)
    if not available:
        raise HTTPException(400, "No supported dataset found under provided `db_path`")

    datasets = []
    old_cwd = os.getcwd()
    os.chdir(project_root)
    try:
        for name in available:
            try:
                df, *_ = load_data(name)
                # 只返回在当前 db_path 下可加载成功的数据集
                datasets.append({"name": name, "columns": list(df.columns)})
            except Exception:
                # 跳过当前 db_path 下不完整/不支持的数据集
                continue
    finally:
        os.chdir(old_cwd)

    if not datasets:
        raise HTTPException(400, "Failed to load dataset metadata")

    return {"datasets": datasets}


# =========================================================
# FilterPilot APIs
# =========================================================
@app.post("/api/iqe/start", response_model=StartResp)
def start(req: StartReq):
    try:
        db_p = Path(req.db_path).expanduser().resolve()
    except Exception:
        return Response(
            content="Invalid db_path",
            status_code=400,
            media_type="text/plain",
        )

    try:
        args = ArgsLite()
        st: SessionState

        # SQLite mode: req.db_path is a sqlite file.
        if db_p.exists() and db_p.is_file():
            try:
                meta = ensure_sqlite_dataset_loaded(db_p, req.table, req.column)
            except Exception as e:
                return Response(
                    content=f"Invalid sqlite table/column: {e}",
                    status_code=400,
                    media_type="text/plain",
                )

            args.dataset = req.table
            st = SessionState(
                session_id=f"sess_{uuid.uuid4().hex[:12]}",
                args=args,
                attribute=meta["attribute"],
                corpus=meta["corpus"],
                bm25_index=meta["bm25"],
                hnsw_index=meta["hnsw"],
                reformat_template="The value is the same as or a type of '{query}'.",
                org_query=req.value.strip(),
                ttl_sec=req.timeLimitSec,
            )
        else:
            # Folder mode (legacy CSV datasets): req.db_path points to dataset folder or its parent.
            dataset_root, project_root = resolve_dataset_root(str(db_p))
            if not dataset_root.exists() or not dataset_root.is_dir():
                return Response(
                    content="Invalid db_path: dataset root not found",
                    status_code=400,
                    media_type="text/plain",
                )

            # Only support these dataset names (load_data current implementation).
            available = set(list_available_datasets(dataset_root))
            if req.table not in available:
                return Response(
                    content=f"Invalid dataset/table: {req.table}",
                    status_code=400,
                    media_type="text/plain",
                )

            old_cwd = os.getcwd()
            os.chdir(project_root)
            try:
                args.dataset = req.table
                meta = ensure_dataset_loaded(args.dataset, req.column)
            except Exception as e:
                return Response(
                    content=f"Invalid dataset/table/column: {e}",
                    status_code=400,
                    media_type="text/plain",
                )
            finally:
                os.chdir(old_cwd)

            st = SessionState(
                session_id=f"sess_{uuid.uuid4().hex[:12]}",
                args=args,
                attribute=meta["attribute"],
                corpus=meta["corpus"],
                bm25_index=meta["bm25"],
                hnsw_index=meta["hnsw"],
                reformat_template="The value is the same as or a type of '{query}'.",
                org_query=req.value.strip(),
                ttl_sec=req.timeLimitSec,
            )

        # LLM-based reformulation
        t0 = time.time()
        samples = meta["samples"]
        init_terms = do_reformulate(args, st.query.org_query, st.attribute, samples)
        t1 = time.time() - t0

        st.query.update_queries_from_generated(init_terms)
        st.query.update_query_scores({q: 2 for q in init_terms})

        with SESS_LOCK:
            SESSIONS[st.session_id] = st

        return StartResp(
            sessionId=st.session_id,
            reformulatedTerms=init_terms,
            t1Sec=round(t1, 3),
        )
    except Exception as e:
        return Response(
            content=f"Failed to start session: {e}",
            status_code=400,
            media_type="text/plain",
        )


@app.post("/api/iqe/iterate", response_model=IterateResp)
def iterate(req: IterateReq):
    st = _get_session_or_404(req.sessionId)
    args = st.args

    # hard stop only
    if st.done:
        return IterateResp(
            t1Sec=0.0,
            reformulatedTerms=[],
            alignedHigh=[],
            alignedPotential=[],
            hnswQueries=[],
            bm25Queries=[],
            summary="Session hard-stopped (budget reached).",
            done=True,
            suggestedStop=st.recommended_stop,
            timeUp=st.time_up,
            checkedTotal=len(st.query.obj_scores),
        )

    st.iteration += 1

    hnsw_queries = st.query.select_hnsw_queries(st.retrieved_info, args.select_query)
    bm25_queries = st.query.select_bm25_queries(
        st.retrieved_info, args.select_query, []
    )

    st.retrieved_info = retrieve_corpus(
        bm25_queries,
        hnsw_queries,
        st.corpus,
        args,
        st.bm25_index,
        st.hnsw_index,
    )

    check_num = min(CHECK_NUM, args.budget - len(st.query.obj_scores))

    obj_to_check, stop_flag = rerank_retrieved_objs(
        st.query, st.retrieved_info, args, check_num
    )

    obj_scores, query_scores = llm_check_retrieved_objs(st.query, obj_to_check, args)

    st.query.update_query_scores(query_scores)

    positives = [o for o, s in obj_scores.items() if s >= 2]
    st.query.update_queries_from_table(positives)

    st.query.update_obj_scores(obj_scores, st.corpus)
    st.query.update_obj_features(st.retrieved_info)

    aligned_high = [o for o, s in query_scores.items() if s >= 1.5]
    aligned_pot = [o for o, s in query_scores.items() if s == 1]

    # ---------- early-stop tracking (per session) ----------
    if len(aligned_high) == 0:
        st.early_stop += 1
    else:
        st.early_stop = 0
    print(f"[DEBUG] early_stop={st.early_stop}, stop_flag={stop_flag}")

    # ---------- soft stop conditions ----------
    if stop_flag == 1 or st.early_stop > 2:
        st.recommended_stop = True

    # ---------- hard stop condition ----------
    if len(st.query.obj_scores) >= args.budget:
        st.done = True

    summary = (
        f"Round {st.iteration}: "
        f"checked={len(obj_scores)}, "
        f"Yes={len(aligned_high)}, Unsure={len(aligned_pot)}"
    )

    if st.recommended_stop:
        summary += " | Algorithm suggests stopping"
    if st.time_up:
        summary += " | Time limit reached"

    st.time_up = time.time() > st.expires_at

    return IterateResp(
        t1Sec=0.0,
        reformulatedTerms=[],
        alignedHigh=aligned_high,
        alignedPotential=aligned_pot,
        hnswQueries=hnsw_queries,
        bm25Queries=bm25_queries,
        summary=summary,
        done=st.done,
        suggestedStop=st.recommended_stop,
        timeUp=st.time_up,
        checkedTotal=len(st.query.obj_scores),
    )
