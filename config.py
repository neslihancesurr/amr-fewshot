"""Central settings. Edit this file, not the scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# --- data -------------------------------------------------------------
GOLD_FILE = ROOT / "data" / "amr_gold.txt"            # all records: '# ::id', '# ::snt', then the Penman graph
TEST_IDS_FILE = ROOT / "data" / "test_ids.txt"        # optional: ids of your 500 test sentences, one per line
ANNOTATION_FILE = ROOT / "data" / "annotations.jsonl" # optional: what the rule-based parser receives (see annotations.py)
SYSTEM_PROMPT_FILE1 = ROOT / "prompts" / "system1.txt"
SYSTEM_PROMPT_FILE2 = ROOT / "prompts" / "system2.txt"
# --- scoring ----------------------------------------------------------
# Put your own smatch.py (and amr.py) next to this config, or point here.
SMATCH_SCRIPT = ROOT / "smatch.py"

# --- Ollama -----------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = "qwen3:8b"     # run 01_check_ollama.py to see what you actually have, or pass --model
NUM_CTX = 8192         # IMPORTANT: Ollama silently truncates prompts longer than this
NUM_PREDICT = 1024     # max output tokens
TEMPERATURE = 0.0
THINK = None           # set to False if a Qwen3-style model prints long reasoning (needs a recent Ollama)

# --- trial run --------------------------------------------------------
NUM_SHOTS = 3
N_TRIAL = 5
SEED = 13
OUT_DIR = ROOT / "outputs"
