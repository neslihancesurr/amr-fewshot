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
```

## Requirements

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Ollama locally:

```bash
ollama serve
```

3. Pull a model if needed, for example:

```bash
ollama pull gemma3:latest
```

4. Update `config.py` as needed, especially the model name, Ollama URL, context size, and prompt settings.

## Typical workflow

This repo is intended to be used in roughly this order:

```bash
python 01_check_ollama.py
python 02_test_prompt.py
python build_fewshot_json.py
python make_test_set.py
python run_fewshot_eval.py
```

The project supports a few experiment styles:

- E1: sentence -> AMR only
- E2: sentence + token annotations -> AMR
- fixed few-shot prompting with demonstrations
- evaluation over a dataset with results written to `outputs/<run_name>/`

## Script-by-script overview

### `01_check_ollama.py`

Checks whether Ollama is running and lists the available local models. It also sends a quick smoke-test prompt to a selected model to confirm that generation works.

Use this first to confirm the server is available and to inspect model names and sizes.

### `02_test_prompt.py`

This is a small interactive prompt test script. It asks for a Turkish sentence, builds a few-shot prompt using the configured system prompt and demonstration examples, sends it to the model, prints the generated AMR graph, and saves the result as a JSON file in `outputs/`.

This is useful for debugging prompt design and checking model behavior before running a larger evaluation.

### `build_fewshot_json.py`

Creates JSON prompt files for few-shot experiments.

It reads a gold AMR file containing only demonstration examples and converts them into a format like:

```json
[
  { "id": "...", "sentence": "...", "amr": "(...)" }
]
```

It can also optionally include token-level annotation data for a richer experiment (E2), where each token includes morphological, semantic, PropBank, FrameNet, and dependency-style information.

### `make_test_set.py`

Creates a clean evaluation test set by removing the few-shot demonstration examples from a larger gold AMR file.

This prevents data leakage: if the same sentence or example IDs appear in both the few-shot set and the test set, they are removed from evaluation.

It can also filter a matching prediction or parser-output file so that the same sentence set and ID order are used for both the gold and system outputs.

### `run_fewshot_eval.py`

This is the main evaluation script. It loops over the test set, builds one prompt per sentence using a system prompt and a fixed few-shot demonstration set, sends the prompt to the model, extracts the AMR graph from the response, validates it, and stores the results.

It writes:

- `raw.jsonl`: one row per sentence, appended as the run progresses
- `pred.txt`: predicted AMR graphs in the project’s AMR file format
- `gold.txt`: matching gold AMR graphs
- `run_config.json`: metadata describing the run

Outputs are stored under `outputs/<run_name>/` so multiple runs can be compared.

### `amr_io.py`

Utility functions for reading and writing AMR files in the project’s format:

```text
# ::id ...
# ::snt ...
<Penman AMR graph>

```

This file defines `AmrRecord` and provides the helper functions used throughout the repo for loading and saving AMR datasets.

### `ollama_client.py`

A small wrapper around the Ollama HTTP API.

It exposes:

- `list_models(base_url)`: lists available local models
- `chat(base_url, model, messages, ...)`: sends a chat prompt and returns the generated text plus token count

This is the interface that connects the repo to the local LLM server.

### `config.py`

The central configuration file. It defines the model, prompt paths, evaluation options, context size, temperature, and general runtime settings.

This is the place to change the model name or experiment parameters without editing every script separately.

## Prompt and data files

### `prompts/`

This folder contains system prompts and few-shot examples:

- `system1.txt`: standard sentence-only prompt
- `system2.txt`: richer prompt that references token annotations and linguistic features
- `fewshot1.json`: few-shot sentence-only examples
- `fewshot2.json`: few-shot examples with richer annotation data

### `data/`

This folder contains AMR datasets in the project’s file format, including gold files and filtered test sets.

The project expects records in a shape similar to:

```text
# ::id example_001
# ::snt Ali kitabı okudu.
(d / read-01
    :ARG0 (p / person)
    :ARG1 (b / book))
```

## General purpose of the repo

In one sentence, this repository is a research-oriented workflow for testing whether a local LLM can perform Turkish AMR parsing with few-shot prompting without relying on cloud APIs.

More concretely, it is meant for:

- experimenting with few-shot AMR generation in Turkish
- comparing sentence-only vs annotation-aware prompting
- evaluating model-generated AMR against gold-standard AMR graphs
- running reproducible local-model experiments with Ollama

## Notes

- This repo assumes a local Ollama server is running.
- The scripts expect AMR data in the correct project format.
- The project is focused on experimentation and evaluation rather than end-user deployment.

## Recommended first run

```bash
python 01_check_ollama.py
python 02_test_prompt.py
```

Once the setup works, continue with the few-shot data preparation and evaluation workflow.

## License

This project does not appear to include a repository-specific license file. Please check the repository contents and add a license if needed before publishing or redistributing the code.
