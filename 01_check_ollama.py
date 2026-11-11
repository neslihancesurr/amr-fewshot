"""Step 1: which models do I have, and does one answer?  Usage: python 01_check_ollama.py [model]"""
import sys
import requests
import config
from ollama_client import list_models, chat

try:
    models = list_models(config.OLLAMA_URL)
except requests.ConnectionError:
    raise SystemExit("Ollama is not running. Start the Ollama app (or run 'ollama serve') and retry.")

print(f"{len(models)} local model(s):")
for m in models:
    d = m.get("details", {})
    print(f"  {m['name']:35s} {m['size']/1e9:5.1f} GB  {d.get('parameter_size','?'):>6}  {d.get('quantization_level','?')}")

model = sys.argv[1] if len(sys.argv) > 1 else (models[0]["name"] if models else None)
if model:
    print(f"\nSmoke test with {model} ...")
    out, ptoks = chat(config.OLLAMA_URL, model,
                      [{"role": "user", "content":"'Ali kitabı okudu.' cümlesini İngilizceye çevir. Sadece çeviriyi yaz."}],
                      num_ctx=2048, num_predict=64, think=config.THINK)
    print("Answer:", out.strip())
