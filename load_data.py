import pandas as pd
import json


def load_data(dataset_name):
    folder = f"dataset/{dataset_name}"
    if dataset_name in ["animal", "chemical_compound", "product"]:
        df = pd.read_csv(f"dataset/{dataset_name}/{dataset_name}.csv")
        query_answer = json.load(open(f"dataset/{dataset_name}/query_answer.json"))
        query_template = f"Find all the {dataset_name} that are " + "{value}"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return df, query_answer, query_template, folder
