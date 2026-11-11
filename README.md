# AMR Few-Shot Experiments with Local LLMs

This repository is a small experimental pipeline for generating Abstract Meaning Representation (AMR) graphs from Turkish sentences using few-shot prompting with local models served through Ollama.

The main goal is to test whether a local LLM can produce valid AMR in Penman notation from a Turkish sentence, optionally using token-level annotation features, with only a few demonstration examples in the prompt.

AMR is a graph-based semantic representation where the meaning of a sentence is encoded as a directed, labeled graph. In this project, the model is prompted to generate one AMR graph per sentence and the outputs are saved for evaluation.

## What this repo does

- Connects to a local Ollama server
- Lists available local models and checks that generation works
- Loads few-shot examples and injects them into prompts
- Builds JSON datasets for AMR experiments
- Creates evaluation test sets while avoiding data leakage
- Runs a few-shot AMR evaluation over a sentence set
- Saves raw model outputs, predictions, and gold AMR files for scoring

## Project structure

```text
.
├── 01_check_ollama.py
├── 02_test_prompt.py
├── amr_io.py
├── build_fewshot_json.py
├── config.py
├── make_test_set.py
├── ollama_client.py
├── run_fewshot_eval.py
├── README.md
├── requirements.txt
├── data/
│   ├── amr_gold_20.txt
│   ├── amr_gold_test.txt
│   ├── gold_500.txt
│   ├── test_500.txt
│   └── ...
├── outputs/
├── prompts/
│   ├── fewshot1.json
│   ├── fewshot2.json
│   ├── fewshot_exp1_sentence_amr.json
│   ├── fewshot_exp2_with_annotations.json
│   ├── system1.txt
│   └── system2.txt
└── ...
