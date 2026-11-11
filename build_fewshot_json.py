"""Build few-shot JSON files for the Turkish AMR LLM experiments.

Set the paths in the SETTINGS block below, then just run this file (click Run in PyCharm).

Input : a gold AMR file that contains ONLY your few-shot sentences
        ('# ::id Frame_Name/0002.train', '# ::snt ...', Penman graph, blank line between records)
Output: experiment 1 -> [{"id", "sentence", "amr"}]
        experiment 2 -> [{"id", "sentence", "tokens": [...], "amr"}]

Experiment 2 reads the annotation file of each sentence from the Dropbox-synced folder:
  <ANNOTATION_ROOT>/<gold id>      e.g.  .../Turkish-Phrase/Abandoned_State/0002.train

Needs amr_io.py (read_amr_file) from your repo in the same folder / on the path.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

from amr_io import read_amr_file

# ============================== SETTINGS ==============================
GOLD_FILE = "data/amr_gold_10.txt"                                     # gold file reduced to the few-shot sentences
OUT_DIR = "prompts"                                          # output folder (created if missing)
ANNOTATION_ROOT = "~/Library/CloudStorage/Dropbox/FrameNet-Examples/Turkish-Phrase"   # real Dropbox location on Mac (not the ~/Dropbox shortcut)
EXPERIMENT = "both"                                               # "1" (sentence + AMR), "2" (+ annotations), or "both"
COMPACT_AMR = False                                               # True = one line per graph (fewer prompt tokens)
EXP1_FILENAME = "fewshot10_e1.json"                  # output names (e.g. "test_e1.json" when building the test set)
EXP2_FILENAME = "fewshot10_e2.json"
# ======================================================================

TOKEN_FIELD_ORDER = ["morphologicalAnalysis", "metaMorphemes", "semantics", "propbank", "framenet"]


def match_key(s):
    """NFC, no whitespace at all (so 'olursun.' and 'olursun .' compare equal)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def annotation_path(gold_id):
    """'Abandoned_State/0002.train' -> <ANNOTATION_ROOT>/Abandoned_State/0002.train
    (the split suffix is part of the file name, so 0002.train and 0002.test never get mixed up)."""
    return Path(ANNOTATION_ROOT).expanduser() / gold_id


# ----------------------------------------------------------------------------- annotation files
def parse_annotation_text(text):
    """'{turkish=..}{morphologicalAnalysis=..}... {turkish=..}...' -> list of token dicts."""
    text = text.strip().lstrip("\ufeff")
    chunks = re.split(r"\}\s+\{(?=turkish=)", text)      # boundary between two tokens
    tokens = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if chunk.startswith("{"):
            chunk = chunk[1:]
        if chunk.endswith("}"):
            chunk = chunk[:-1]
        fields = {}
        for field in re.split(r"\}\{(?=[A-Za-z]+=)", chunk):  # boundary between two fields
            key, _, value = field.partition("=")
            fields[key.strip()] = value
        tokens.append(build_token(i + 1, fields))
    return tokens


def build_token(idx, f):
    tok = {"id": idx, "form": f.get("turkish")}
    for key in TOKEN_FIELD_ORDER:
        tok[key] = f.get(key) or None                    # missing (e.g. punctuation) -> null
    ud = f.get("universalDependency")
    if ud:
        head, _, rel = ud.partition("$")
        tok["universalDependency"] = {"head": int(head) if head.isdigit() else head, "relation": rel}
    else:
        tok["universalDependency"] = None
    for key, value in f.items():                         # keep anything unexpected instead of dropping it
        if key not in tok and key not in ("turkish", "universalDependency"):
            tok[key] = value
    return tok


# ----------------------------------------------------------------------------- main
def main():
    records = read_amr_file(GOLD_FILE)
    print(f"{len(records)} records read from {GOLD_FILE}")

    problems = []
    ids = [r.id for r in records]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"id appears more than once in the gold file: {dup}")

    def amr_of(rec):
        return re.sub(r"\s+", " ", rec.graph) if COMPACT_AMR else rec.graph

    want2 = EXPERIMENT in ("2", "both")
    exp1, exp2 = [], []
    for rec in records:
        exp1.append({"id": rec.id, "sentence": rec.snt, "amr": amr_of(rec)})
        if not want2:
            continue
        p = annotation_path(rec.id)
        if not p.exists():
            problems.append(f"annotation file not found: {p}")
            continue
        try:
            tokens = parse_annotation_text(p.read_text(encoding="utf-8-sig"))
        except Exception as e:
            problems.append(f"{p}: could not parse ({e})")
            continue
        forms = "".join(t["form"] or "" for t in tokens)
        if match_key(forms) != match_key(rec.snt):
            print(f"WARNING {rec.id}: token forms do not spell the gold sentence - wrong file?\n"
                  f"   tokens  : {' '.join(t['form'] or '' for t in tokens)}\n   sentence: {rec.snt}")
        exp2.append({"id": rec.id, "sentence": rec.snt, "tokens": tokens, "amr": amr_of(rec)})

    if problems:
        print("\nNothing written - fix these first:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if EXPERIMENT in ("1", "both"):
        p = out / EXP1_FILENAME
        p.write_text(json.dumps(exp1, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {p}  ({len(exp1)} examples)")
    if want2:
        p = out / EXP2_FILENAME
        p.write_text(json.dumps(exp2, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {p}  ({len(exp2)} examples)")


if __name__ == "__main__":
    main()