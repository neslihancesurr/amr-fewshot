"""Make the LLM test set: your gold file minus the few-shot demonstration sentences.

Set the paths below and click Run. Nothing is modified in place; a new file is written.

Removes a record if
  - its id is one of the first NUM_TO_REMOVE demonstrations in DEMO_FILE, or
  - its sentence text is identical to one of those demonstrations (same sentence under another id would
    still be a leak).

Optionally filters other AMR files (e.g. the rule-based parser's output) to exactly the same
sentences, in the same order, so the LLM and the rule-based parser are scored on one identical test set.
"""
import json
import re
import unicodedata
from pathlib import Path

from amr_io import read_amr_file, write_amr_file

# ============================== SETTINGS ==============================
GOLD_FILE = "data/gold_500.txt"                  # the full gold file (with the few-shot sentences still in it)
DEMO_FILE = "prompts/fewshot1.json"         # the few-shot JSON (ids must be the gold ids)
NUM_TO_REMOVE = 10                          # = NUM_SHOTS you use; set to 20 if you will ever run 20 shots; None = all in the file
OUT_FILE = "data/amr_gold_test.txt"              # the new test gold file
ALSO_FILTER = ["data/test_500.txt"]                            # e.g. ["rule_based_output.txt"] -> written as rule_based_output_test.txt
# ======================================================================


def key(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def main():
    gold = read_amr_file(GOLD_FILE)
    demos = json.load(open(DEMO_FILE, encoding="utf-8"))
    demos = demos if NUM_TO_REMOVE is None else demos[:NUM_TO_REMOVE]
    drop_ids = {d["id"] for d in demos}
    drop_snts = {key(d["sentence"]): d["id"] for d in demos}

    gold_ids = {r.id for r in gold}
    not_found = sorted(drop_ids - gold_ids)
    if not_found:
        print(f"WARNING: {len(not_found)} demonstration id(s) are not in the gold file (already removed, or id mismatch?):")
        for i in not_found:
            print("   ", i)

    kept = []
    for r in gold:
        if r.id in drop_ids:
            continue
        if key(r.snt) in drop_snts:
            print(f"Also removed {r.id}: same sentence as demonstration {drop_snts[key(r.snt)]}")
            continue
        kept.append(r)

    write_amr_file(OUT_FILE, kept)
    print(f"gold records: {len(gold)}   demonstrations removed: {len(drop_ids & gold_ids)}   "
          f"kept: {len(kept)}   -> {OUT_FILE}")

    for path in ALSO_FILTER:
        recs = {r.id: r for r in read_amr_file(path)}
        if not recs:
            print(f"WARNING: no records read from {path}")
            continue
        missing = [r.id for r in kept if r.id not in recs]
        if missing:
            print(f"WARNING: {len(missing)} test ids are missing in {path}, e.g. {missing[:3]}")
        out = Path(path).with_name(Path(path).stem + "_test" + Path(path).suffix)
        write_amr_file(out, [recs[r.id] for r in kept if r.id in recs])
        print(f"{path} -> {out}  ({sum(r.id in recs for r in kept)} records, gold order)")


if __name__ == "__main__":
    main()