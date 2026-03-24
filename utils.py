import sqlite3
import re
import numpy as np
import json


def parse_json(ans: str) -> dict:
    ans = ans.strip()
    start_idx = ans.find("{")
    end_idx = ans.rfind("}") + 1
    ans = ans[start_idx:end_idx]
    return json.loads(ans)


def execute_sql(sql: str, db: str) -> list:
    """
    Execute a SQL command and return the results.
    Args:
        sql (str): The SQL command to execute.
        db (str): The database string.
    Returns:
        list: The results of the SQL command.
    """
    # This is a placeholder function. Replace with actual database execution logic.
    # For example, using sqlite3 or any other database connector.
    conn = sqlite3.connect(f"data/db/{db}.db")
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results


def sample_table(db: str, table: str, num_samples=5) -> dict:
    """
    Sample the table to get column names and sample values.
    Args:
        table (str): The table string.
        num_samples (int): The number of samples to extract.
    Returns:
        dict: A dictionary mapping column names to sample values.
    """
    # This is a placeholder function. Replace with actual sampling logic.
    # For example, if the table is a CSV file, you might want to read it and sample rows.
    sql = f"PRAGMA table_info({table});"
    # Execute the SQL command to get column names
    results = execute_sql(sql, db)
    columns = [row[1] for row in results]
    # Sample values from the table
    col_vals = {}
    for col in columns:
        sql = f"SELECT distinct {col} FROM {table} WHERE {col} IS NOT NULL LIMIT {num_samples};"
        values = execute_sql(sql)
        col_vals[col] = [row[0] for row in values]
    return col_vals


def parse_response(ans, pattern):
    ans = str(ans)
    pattern = f"<{pattern}>(.*?)</{pattern}>"
    match = re.search(rf"{pattern}", ans, re.DOTALL)
    if match:
        response = match.group(1).strip()
        return response
    return "unknown"


def cal_ndcg(sorted_ids, pos_ids) -> float:
    if len(sorted_ids) == 0:
        return 0
    if len(pos_ids) == 0:
        return 0
    dcg = 0
    for i, id in enumerate(sorted_ids):
        if id in pos_ids:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i, id in enumerate(pos_ids):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg


def uct_selection(credits: dict, visits: dict) -> list:
    query_scores = {}
    total_visits = sum(visits.values())
    selected_queries = []
    for q, n in visits.items():
        w = credits.get(q, 0)
        if n < 1:
            if w >= 0:
                selected_queries.append(q)
            continue
        score = w + np.sqrt(1e-4 * np.log(total_visits) / n)
        query_scores[q] = score
        print(f"q: {q}, score: {score:.4f}, w: {w:.4f}, n: {n}")
    print("=================")
    query_scores = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)
    return [q for q, _ in query_scores][:5] + selected_queries
