# Few-shot Turkish AMR with local LLMs (Ollama)

1. PyCharm: File > Open this folder; create a venv; `pip install -r requirements.txt`.
2. Copy your gold file to `data/amr_gold.txt`, your 500 test ids to `data/test_ids.txt` (optional),
   the parser-input annotations to `data/annotations.jsonl` (format in `annotations.py`),
   and your `smatch.py` + `amr.py` next to `config.py`.
3. Edit the conventions block in `prompts/system.txt`.
4. Run in order: `01_check_ollama.py` -> `02_trial_run.py --model <name>` -> `03_score.py <run_folder>`.
