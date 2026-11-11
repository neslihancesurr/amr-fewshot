import requests


def list_models(base_url):
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def chat(base_url, model, messages, num_ctx, num_predict, temperature=0.0, think=None, seed=0):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": num_predict, "seed": seed},
    }
    if think is not None:
        payload["think"] = think
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
    except requests.ConnectionError:
        raise SystemExit(f"Cannot reach Ollama at {base_url}. Is the Ollama app / 'ollama serve' running?")
    if r.status_code != 200:
        raise SystemExit(f"Ollama error {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["message"]["content"], data.get("prompt_eval_count")
