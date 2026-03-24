# from appl import ppl, gen, SystemMessage, convo, records, SystemRole
# from appl.compositor import Tagged, NumberedList, DashList
from llm_check import run_inference
from utils import parse_json


def reformulate_synonym_prompt(cond: str, col: str, value: list, history: list = []):
    system_prompt = "You are an expert database assistant specializing in query expansion and semantic matching."
    user_prompt = f"""
Your task is to generate semantically equivalent search terms for the 'Original Query Term'.

Instructions:
1. Analyze the query's core concept and identify conceptual synonyms, aliases, abbreviations and alternative terminology
2. Ensure the generated search terms are meaningfully diversified and avoid repeating identical or overly similar terms. Generate 1 to 5 values, depending on the availability of appropriate alternatives.
3. Preserve the non-linguistic characters (e.g., symbols, numbers, formatting) in the original query term.
4. Use the Sample Values as format references to ensure consistency with database conventions.
5. If fewer than 5 meaningful and relevant terms can be generated, only output the appropriate number of terms.

Input:
Original Query Term: {cond}
Column: {col}
Sample Values: {value}

Output:
Directly output the generated search terms, separated by |, without any additional text.
"""
    return system_prompt, user_prompt


def reformulate_narrower_prompt(cond: str, col: str, value: list, history: list = []):
    system_prompt = "You are an expert database assistant specializing in query expansion and semantic matching."
    user_prompt = f"""
Your task is to generate search terms that are more specific instances/subtypes of the original query term.

Instructions:
1. Analyze the query's core concept and identify more specific instances/subtypes of the original query term
2. Ensure the generated search terms are meaningfully diversified and avoid repeating identical or overly similar terms. Generate 1 to 5 values, depending on the availability of appropriate alternatives.
3. Preserve the non-linguistic characters (e.g., symbols, numbers, formatting) in the original query term.
4. Use the Sample Values as format references to ensure consistency with database conventions.
5. If fewer than 5 meaningful and relevant terms can be generated, only output the appropriate number of terms.

Input:
Original Query Term: {cond}
Column: {col}
Sample Values: {value}

Output:
Directly output the generated search terms, separated by |, without any additional text.
"""
    return system_prompt, user_prompt


def identify_identifier_prompt(cond: str):
    # system_prompt = "You are a highly skilled assistant tasked with identifying the identifier in a user's query."
    system_prompt = None
    user_prompt = f"""Identify all continuous sequences of words or characters in the given query that do not form standard words in natural language. This includes, but is not limited to, random alphanumeric strings, meaningless character combinations, abbreviations, or codes that lack inherent semantic meaning in natural language.

Query: {cond}
Output ONLY the identified sequence, or "None" if no such sequence is found.
"""
    return system_prompt, user_prompt


def reformulate_identifier_prompt(
    org_id: str, col: str, value: list, history: list = []
):
    system_prompt = None
    user_prompt = f"""Generate 1-5 stylistic variations of the "Original Sequence" by reformatting it to mimic the formatting style (e.g., delimiter placement and spacing) of non-linguistic sequences in the "Sample Values."
The reformatted sequences must contain exactly the same alphanumeric characters as the "Original Sequence" (no additions or removals).

Input:
Original Sequence: {org_id}
Column: {col}
Sample Values: {value}

Directly output the generated reformatted terms, separated by |, without any additional text.
"""
    return system_prompt, user_prompt


def get_reformulate_prompt(
    cond: str, col: str, value: list, history: list = [], reform_type: str = "zero-shot"
):
    system_prompt = "You are an expert database assistant specializing in query expansion and semantic matching."
    user_prompt = f"""Your task is to generate semantically related search terms based on:
1. A user's query input (which may contain synonyms, abbreviations, partial matches, or hierarchical concepts)
2. A list of sample values from a database column (to maintain format consistency)

Instructions:
1. Analyze the query's core concept and identify:
    - Conceptual synonyms, abbreviations, and alternative terminology
    - Narrower terms (more specific instances/subtypes)
2. Use the example values as format references to maintain consistency with database conventions
3. Generate 2-10 expanded and diversified search terms
4. Exclude exact duplicates of the original query term and historical generated values if any
5. Prioritize terms that would actually appear in database records
"""
    if reform_type == "few-shot":
        user_prompt += f"""
Examples:
Input:
Original Query Term: coffee
Column: drink
Sample Values: ["cappuccino", "black tea", "water", "orange juice", "soda"]
Output: cappuccino | espresso | mocha | latte | americano | drip coffee | affogato

Input:
Original Query Term: Cantonese dim sum
Column: food
Sample Values: ["spaghetti carbonara", "chicken tikka masala", "beef bourguignon", "miso glazed salmon", "vegetable paella"]
Output: ["shrimp dumpling", "barbecue pork bun", "rice noodle roll", "chicken feet", "turnip cake", "taro dumpling", "egg tart"]

Now you are given the following input:
"""

    user_prompt += f"""
Input:
Original Query Term: {cond}
Column: {col}
Sample Values: {value}

"""
    if len(history) > 0:
        history_str = " | ".join([f"{h}" for h in history])
        user_prompt += f"Previously Generated Values:\n{history_str}\n"
    if reform_type == "cot":
        user_prompt += f"""Your output should follow this format:
<think>Your thinking process for generating the search terms</think>
<output>The generated search terms separated by ' | '</output>
"""
    else:
        user_prompt += "Output: please directly output the generated search terms separated by ' | ' without any other text. If no more terms can be generated, respond with 'None'."
    return system_prompt, user_prompt


def reformulate(cond: str, col: str, value: list, non_linguistic_value: list) -> str:
    """
    Reformulate the condition based on the column and value.

    Args:
        cond (str): The original condition.
        col (str): The column name.
        value (list): The list of values.

    Returns:
        str: The reformulated condition.
    """
    sample_vals = value
    print(f"Query Condition: {cond}")
    system_prompt, user_prompt = identify_identifier_prompt(cond)
    ans = run_inference([user_prompt])
    org_id = ans[0].strip()
    print(f"LLM Identify Identifier: {org_id}")
    non_linguistic_values = []
    if org_id.lower() != "none":
        if org_id not in cond:
            words = org_id.split(" ")
            start_idx = cond.find(words[0])
            end_idx = cond.find(words[-1]) + len(words[-1]) + 1
            org_id = cond[start_idx:end_idx]
            print(f"LLM Identify Identifier (corrected): {org_id}")
        if len(org_id.strip()) > 0:
            non_linguistic_values.append(org_id)
        else:
            org_id = None
    else:
        org_id = None
    system_prompt1, user_prompt1 = reformulate_synonym_prompt(cond, col, value)
    system_prompt2, user_prompt2 = reformulate_narrower_prompt(cond, col, value)
    system_prompts = [system_prompt1, system_prompt2]
    user_prompts = [user_prompt1, user_prompt2]
    if org_id is not None:
        system_prompt3, user_prompt3 = reformulate_identifier_prompt(
            org_id, col, non_linguistic_value
        )
        system_prompts.append(system_prompt3)
        user_prompts.append(user_prompt3)
    ans_all = run_inference(user_prompts, system_prompts=system_prompts)
    print(f"LLM Reformulate (raw results): {ans_all}")
    ans_synonym, ans_narrower = ans_all[0], ans_all[1]
    if org_id is not None:
        ans_non_linguistic = ans_all[2]
        values = ans_non_linguistic.split("|")
        values = [v.split("\n")[0].strip() for v in values if len(v.strip()) > 0]
        non_linguistic_values.extend(values)
        alphanumeric_chars = [c for c in org_id if c.isalnum()]
        org_id_alphanumeric = "".join(alphanumeric_chars)
        non_linguistic_values.append(org_id_alphanumeric)
        non_linguistic_values.append(org_id)
        non_linguistic_values = list(set(non_linguistic_values))
        print(f"LLM Reformulate Non-linguistic: {non_linguistic_values}")
        non_linguistic_values = [
            cond.replace(org_id, v) + " " + " ".join([v] * 10)
            for v in non_linguistic_values
        ]
    if ans_synonym.strip().lower() != "none":
        synonym_values = ans_synonym.split("|")
    else:
        synonym_values = []
    if ans_narrower.strip().lower() != "none":
        narrower_values = ans_narrower.split("|")
    else:
        narrower_values = []
    linguistic_values = [
        v.split("\n")[0].strip()
        for v in synonym_values + narrower_values
        if len(v.split("\n")[0].strip()) > 0
    ]
    linguistic_values = list(set(linguistic_values))
    print(f"LLM Reformulate Linguistic: {linguistic_values}")
    return linguistic_values, non_linguistic_values


def reformulate_old(
    cond: str, col: str, value: list, history: list = [], reform_type: str = "zero-shot"
) -> str:
    """
    Reformulate the condition based on the column and value.

    Args:
        cond (str): The original condition.
        col (str): The column name.
        value (list): The list of values.

    Returns:
        str: The reformulated condition.
    """
    # sample_vals = np.random.choice(value, 5, replace=False)
    sample_vals = value
    # if len(sample_vals) > 5:
    #     sample_vals = np.random.choice(sample_vals, 5, replace=False)
    print(f"Query Condition: {cond}")
    system_prompt, user_prompt = get_reformulate_prompt(
        cond, col, sample_vals, history, reform_type
    )
    # ans = gen_reformulate(cond, col, sample_vals, history=history)
    ans = run_inference([user_prompt], system_prompts=[system_prompt])
    ans = ans[0]
    print(f"LLM Reformulate: {ans}")
    if reform_type == "cot":
        ans = ans.split("<output>")[1].split("</output>")[0].strip()
    values = str(ans).split("|")
    if "None" in values:
        return []
    values = [v.split("\n")[0].strip() for v in values if len(v.strip()) > 0]
    return values


def reformulate_simple(
    cond: str, col: str, value: list, non_linguistic_value: list
) -> str:
    system_prompt = "You are an expert database assistant specializing in query expansion and semantic matching."
    user_prompt = f"""Your task is to generate 2-10 expanded and diversified search terms for the "Original Query Term". Use the example values as format references to maintain consistency with database conventions.

Input:
Original Query Term: {cond}
Column: {col}
Sample Values: {value}

Output: please directly output the generated search terms separated by ' | ' without any other text. If no more terms can be generated, respond with 'None'.
"""
    ans = run_inference([user_prompt], system_prompts=[system_prompt])
    ans = ans[0]
    print(f"LLM Reformulate: {ans}")
    values = str(ans).split("|")
    if "None" in values:
        return []
    values = [v.split("\n")[0].strip() for v in values if len(v.strip()) > 0]
    return values, []


# @ppl
# def gen_score(condition: str, query_list: list):
#     SystemMessage("You are an expert in scoring the query words in the query list.")

#     "Given the condition and a list of values, assign each value a score from 0 to 2 based on whether it satisfies the condition:"
#     with DashList():
#         "0: Not satisfies the condition"
#         "1: Not sure"
#         "2: Satisfies the condition"

#     "Return only a JSON object in this format: {'value1': score1, 'value2': score2, ...}."
#     "Do not include any additional text or explanation."
#     f"Condition: {condition}"
#     f"Values: {query_list}"
#     "JSON Output:"
#     return gen()


def score_query(condition: str, query_list: list) -> list:
    """
    Score the query words in the query list.
    """
    # check each query in the query list: zero-shot check if the query satisfies the condition
    if len(query_list) == 0:
        return {}
    prompts = []
    for q in query_list:
        prompt = f"Please check if the value '{q}' satisfies the condition '{condition}'. Directly answer with 'Yes', 'No' or 'Unsure' without any other text."
        prompts.append(prompt)
    results = run_inference(prompts, max_tokens=1)
    score_dict = {}
    for q, result in zip(query_list, results):
        if result.strip().lower().startswith("yes"):
            score_dict[q] = 2
        elif result.strip().lower().startswith("no"):
            score_dict[q] = 0
        else:
            score_dict[q] = 1
    return score_dict
