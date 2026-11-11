"""Run the few-shot experiment over the whole test set, one sentence at a time.

Every sentence is an independent request: system prompt + the same fixed demonstrations + that one sentence.
Nothing from earlier test sentences is carried over.

Writes to outputs/<run name>/ :
  raw.jsonl       one line per sentence, appended as it goes -> safe to stop / crash / restart,
                  finished ids are skipped on the next run
  pred.txt        predictions in '# ::id / # ::snt / graph' format   -> give this to the bootstrap / Smatch code
  gold.txt        the matching gold graphs, same sentences, same order
  run_config.json what produced this run

Test file: a JSON list made with build_fewshot_json.py from your 500-sentence gold file
  E1: [{"id", "sentence", "amr"}]            E2: [{"id", "sentence", "tokens": [...], "amr"}]
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import config
from amr_io import AmrRecord, write_amr_file
from ollama_client import chat

try:
    import penman
except ImportError:
    penman = None

HERE = Path(__file__).resolve().parent

# ================================ SETTINGS ================================
MODEL = "gemma3:latest"
EXPERIMENT = "E1"          # "E1": sentence -> AMR    |    "E2": sentence + token annotations -> AMR
NUM_SHOTS = 10
NUM_CTX = 8192
NUM_PREDICT = 1024
TEMPERATURE = 0.0
SEED = 0
LIMIT = 5               # e.g. 5 = only the first 5 test sentences (smoke test); None = all

SYSTEM_FILE = HERE / "prompts" / "system1.txt"
DEMO_FILE = HERE / "prompts" / "fewshot1.json"      # E2: use the E2 demonstrations file
TEST_FILE = HERE / "data" / "test_e1.json"          # E2: use the E2 test file (with tokens)
OUTPUT_DIR = HERE / "outputs"
# ==========================================================================

PLACEHOLDER = "(a / amr-empty)"   # written to pred.txt when the model gives no valid graph, so Smatch stays aligned


# ----------------------------------------------------------------------------- prompt
def load_system_prompt():
    text = SYSTEM_FILE.read_text(encoding="utf-8")
    return text.replace("[DEMONSTRATIONS WILL BE INSERTED HERE]", "").strip()


def format_input(example):
    if EXPERIMENT == "E1":
        return example["sentence"]
    # E2: sentence + token annotations, compact JSON (same format for demos and test sentences)
    return json.dumps({"sentence": example["sentence"], "tokens": example["tokens"]},
                      ensure_ascii=False, separators=(",", ":"))


def build_messages(system_prompt, demonstrations, example):
    messages = [{"role": "system", "content": system_prompt}]
    for demo in demonstrations:
        messages.append({"role": "user", "content": format_input(demo)})
        messages.append({"role": "assistant", "content": demo["amr"]})
    messages.append({"role": "user", "content": format_input(example)})
    return messages


# ----------------------------------------------------------------------------- output cleaning
def extract_graph(raw):
    """Model output -> just the Penman graph (drops <think> blocks, code fences, chatter around it)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    raw = re.sub(r"```[a-zA-Z]*", "", raw)
    start = raw.find("(")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "(":
            depth += 1
        elif raw[i] == ")":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1].strip()
    return raw[start:].strip()          # unbalanced -> fails validation


def is_valid(graph):
    if not graph:
        return False
    if penman is None:                  # pip install penman for a real check
        return graph.count("(") == graph.count(")")
    try:
        penman.decode(graph)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- run bookkeeping
def run_name():
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return "_".join([safe(MODEL), EXPERIMENT, f"{NUM_SHOTS}shot", SYSTEM_FILE.stem, DEMO_FILE.stem])


def load_rows(raw_path):
    rows = {}
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["id"]] = row
    return rows


def write_outputs(run_dir, test, rows):
    gold, pred = [], []
    for ex in test:                      # test-file order -> pred and gold are always aligned
        row = rows.get(ex["id"])
        if row is None:
            continue
        gold.append(AmrRecord(ex["id"], ex["sentence"], ex["amr"]))
        pred.append(AmrRecord(ex["id"], ex["sentence"], row["graph"] if row["valid"] else PLACEHOLDER))
    write_amr_file(run_dir / "pred.txt", pred)
    write_amr_file(run_dir / "gold.txt", gold)
    return len(pred)


# ----------------------------------------------------------------------------- main
def main():
    system_prompt = load_system_prompt()
    with open(DEMO_FILE, encoding="utf-8") as f:
        demonstrations = json.load(f)[:NUM_SHOTS]
    if len(demonstrations) < NUM_SHOTS:
        raise ValueError(f"{DEMO_FILE.name} has only {len(demonstrations)} examples, NUM_SHOTS = {NUM_SHOTS}")
    with open(TEST_FILE, encoding="utf-8") as f:
        test = json.load(f)
    if LIMIT:
        test = test[:LIMIT]

    overlap = {d["id"] for d in demonstrations} & {t["id"] for t in test}

    if overlap:
        raise ValueError(f"These demonstration ids are also in the test set (data leak): {sorted(overlap)}")

    run_dir = OUTPUT_DIR / run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    rows = load_rows(raw_path)
    todo = [t for t in test if t["id"] not in rows]

    (run_dir / "run_config.json").write_text(json.dumps({
        "model": MODEL, "experiment": EXPERIMENT, "num_shots": NUM_SHOTS,
        "demonstration_ids": [d["id"] for d in demonstrations],
        "system_prompt_file": SYSTEM_FILE.name, "demo_file": DEMO_FILE.name, "test_file": TEST_FILE.name,
        "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": TEMPERATURE, "seed": SEED,
        "last_started": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run: {run_dir.name}")
    print(f"Test sentences: {len(test)}   already done: {len(test) - len(todo)}   to do: {len(todo)}")

    t_start = time.time()
    with open(raw_path, "a", encoding="utf-8") as raw_file:
        for i, ex in enumerate(todo, 1):
            messages = build_messages(system_prompt, demonstrations, ex)
            t0 = time.time()
            output, prompt_tokens = chat(
                config.OLLAMA_URL, MODEL, messages,
                num_ctx=NUM_CTX, num_predict=NUM_PREDICT,
                temperature=TEMPERATURE, think=getattr(config, "THINK", False), seed=SEED)
            seconds = time.time() - t0
            graph = extract_graph(output)
            valid = is_valid(graph)
            row = {"id": ex["id"], "sentence": ex["sentence"], "raw_output": output, "graph": graph,
                   "valid": valid, "prompt_tokens": prompt_tokens, "seconds": round(seconds, 2)}
            rows[ex["id"]] = row
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_file.flush()                                     # nothing is lost if the run is interrupted

            warn = ""
            if prompt_tokens and prompt_tokens >= NUM_CTX - NUM_PREDICT:
                warn = "  !! prompt may be truncated: raise NUM_CTX or lower NUM_SHOTS"
            eta = (time.time() - t_start) / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {ex['id']}  valid={valid}  prompt_tokens={prompt_tokens}  "
                  f"{seconds:.1f}s  ETA {eta:.0f} min{warn}")

    n = write_outputs(run_dir, test, rows)
    n_valid = sum(rows[t["id"]]["valid"] for t in test if t["id"] in rows)
    print(f"\nDone. {n} sentences written, {n_valid} valid Penman ({100 * n_valid / max(n, 1):.1f}%).")
    print(f"pred: {run_dir / 'pred.txt'}\ngold: {run_dir / 'gold.txt'}")


if __name__ == "__main__":
    main()