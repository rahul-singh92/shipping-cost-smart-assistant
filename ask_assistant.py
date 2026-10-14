import sys
import pandas as pd
import requests
from src.rag_engine import get_api_key, get_active_model, GROQ_API_URL


def ask_question(question, results_csv="output/shipping_cost_anomalies.csv"):
    df = pd.read_csv(results_csv).fillna("")
    findings_context = df.to_csv(index=False)

    api_key = get_api_key()
    model_name = get_active_model(api_key)

    system_prompt = (
        "You are the FreightTiger Shipping Cost Smart Assistant. Answer the user's question "
        "in clear, plain English using ONLY the provided anomaly findings table. "
        "Cite exact routes, weeks, percentages, verdicts, and note IDs where relevant. "
        "Do not invent facts outside the table."
    )

    user_prompt = f"""ANOMALY FINDINGS TABLE:
{findings_context}

USER QUESTION:
{question}"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 400,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"\nQ: {q}\n")
        print(f"A: {ask_question(q)}\n")
    else:
        print("=== FreightTiger Smart Assistant Q&A (type 'exit' to quit) ===")
        while True:
            q = input("\nAsk a question about shipping costs: ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                break
            if q:
                print(f"\nAssistant:\n{ask_question(q)}")