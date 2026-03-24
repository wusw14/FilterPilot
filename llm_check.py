import openai
from typing import List
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
import os


load_dotenv(".env")
model_path = os.getenv("MODEL_PATH")
port = os.getenv("PORT")
api_key = os.getenv("API_KEY")

client = openai.OpenAI(
    base_url=f"http://localhost:{port}/v1",  # vLLM server address
    api_key=api_key,
)


def process_single_prompt(cond: str, col: str, val: str) -> str:
    # replace the placeholder {col} with val
    statement = cond.replace(f"{{{col}}}", val)
    content = f"Please check if the statement '{statement}' is correct. Respond only with 'True' or 'False'."
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": content},
    # ]
    # return messages
    return content


def llm_check(
    query: str, corpus: list, template: str, checked_results={}, max_tokens=1024
) -> list:
    prompts = []
    corpus_new = []
    corpus = list(set(corpus))
    for value in corpus:
        if value in checked_results:
            continue
        corpus_new.append(value)
        prompt = template.format(value=value, query=query)
        # prompt += " Please directly answer with 'Yes' or 'No'."
        prompts.append(prompt)
    if len(prompts) == 0:
        return {}
    results = run_inference(prompts, max_tokens)
    obj_scores = {}
    filtered_vals = []
    for val, result in zip(corpus_new, results):
        if result.strip().lower().startswith("yes"):
            obj_scores[val] = 2
        elif result.strip().lower().startswith("unsure"):
            obj_scores[val] = 1
        else:
            obj_scores[val] = 0
            # filtered_vals.append(val)
    # return filtered_vals
    return obj_scores


def run_inference(
    prompts: List[str],
    max_tokens=1024,
    system_prompts: list = None,
    temperature=0.0,
) -> List[str]:

    def generate_completion(prompt, system_prompt=None):
        messages = [
            {"role": "user", "content": prompt},
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=42 if temperature == 0 else None,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content

    # def generate_completion(prompt):
    #     ans = appl_gen(prompt, max_tokens)
    #     ans = str(ans).strip()
    #     return ans

    with ThreadPoolExecutor(max_workers=min(100, len(prompts))) as executor:
        if system_prompts:
            completions = list(
                executor.map(generate_completion, prompts, system_prompts)
            )
        else:
            completions = list(executor.map(generate_completion, prompts))
    return completions
