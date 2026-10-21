# REDACT — Red-team Dataset Automation & Construction Toolkit

[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://centresecuriteia.github.io/REDACT/)
[![Datasets](https://img.shields.io/badge/🤗%20datasets-7-yellow.svg)](https://huggingface.co/centrepourlasecuriteia)
[![Sister project: BELLS-O](https://img.shields.io/badge/sister%20project-BELLS--O-3f51b5.svg)](https://centresecuriteia.github.io/BELLS-O/)

REDACT builds red-teaming datasets for content moderation and jailbreak
research. Give it a taxonomy of harm categories and a recipe: it generates
samples for every category, checks each one with a second model, generates
model responses, augments the samples with jailbreak techniques and natural
variations, and merges the accepted rows into one CSV for training, or into
three flat files for evaluation. Every stage saves as it goes and resumes
where it stopped. It is for AI safety researchers and
engineers who build or evaluate content-moderation filters and jailbreak
guardrails, and is built for defensive research. It is developed by
[CeSIA — Centre pour la Sécurité de l'IA](https://www.securite-ia.fr/), and
grew out of the datasets of its sister project
[BELLS-O](https://centresecuriteia.github.io/BELLS-O/), a benchmark of AI
supervision systems.

📖 **Full documentation:** <https://centresecuriteia.github.io/REDACT/>

## How it works

A run is an ordered list of stages. Each is one function you can also call
on its own:

1. **constitution**: a strong model writes, per taxonomy category, sample
   descriptions at four severities: harmful, dual-use harmful, dual-use
   benign, benign (an eval run keeps the two clear ends); and benign ones
   per category of a benign taxonomy.
2. **inputs**: an uncensored model turns each description into prompts, and
   a checker model accepts or rejects every one.
3. **outputs**: k checked model responses per accepted input (2 in a training run), each in an answer mode, harmful or helpful.
4. **paraphrase**: checked rewordings of the inputs (and, opt-in, the outputs),
   to remove the generating model's fingerprint.
5. **jailbreaks**: attacks drawn from 183 techniques in 25 families, dealt
   evenly across categories, and natural variations of the inputs.
6. **build**: the accepted rows of every stage, merged into
   `complete_dataset.csv` for training, or into three flat files for
   evaluation (`eval_inputs.csv`, `eval_outputs.csv`, `eval_jailbreaks.csv`).

Models are hosted APIs or local vLLM engines, planned onto the GPUs stage by
stage. Prompts are JSON files that a run can override one by one.
[docs/overview.md](docs/overview.md) draws the files each stage reads and
writes.

## Install

Python 3.11 or later.

```bash
git clone https://github.com/CentreSecuriteIA/REDACT.git
cd REDACT
pip install -e .            # hosted models (Venice, Anthropic, OpenRouter)
pip install -e ".[vllm]"    # + local models through vLLM
```

Put the API keys (`VENICE_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`) in a `.env` file at the repository root. The other
extras are in [docs/running.md](docs/running.md#install).

## Quick start

Run the example recipe, all six stages, into `./runs/first`:

```bash
python scripts/run.py src/redact/configs/runs/training_example.json --data-dir ./runs/first
```

The merged dataset lands in `./runs/first/Datasets/complete_dataset.csv`.
The `paraphrase` stage loads a 12B model, and the `jailbreaks` stage the 27B
TranslateGemma translator, on a local GPU. To run on hosted models only, take
`paraphrase` out of the recipe's `stages` and set `"translation": false` in
`augmentations.adversarial` (or name the hosted `deepseek-v3.2` as the
`translation` model; [docs/running.md](docs/running.md#choose-models)).
`eval_example.json` makes a small eval set the same way, as `eval_inputs.csv`,
`eval_outputs.csv` and
`eval_jailbreaks.csv`. A recipe's `dataset_type`, `training` or `eval`, picks
the defaults ([docs/configuration.md](docs/configuration.md#dataset-types)).

Or call a stage from Python:

```python
from redact import generate_constitution, generate_inputs

constitution = generate_constitution(data_dir="./runs/first", num_taxonomy_categories=3)
inputs = generate_inputs(data_dir="./runs/first", constitution_df=constitution, samples_per_entry=3)
```

## Documentation

- [Documentation site](https://centresecuriteia.github.io/REDACT/), with the
  [API reference](https://centresecuriteia.github.io/REDACT/api/)
- [Overview](docs/overview.md): the stages, the files they pass on, one run
  end to end
- [Running a pipeline](docs/running.md): recipes, models, the GPU planner,
  resume, cost, common failures
- [Configuration](docs/configuration.md): every recipe key, params key and
  registry field
- [Jailbreak techniques](docs/techniques.md): every technique by layer and
  family
- [Extending](docs/extending.md): add a model, a prompt, a technique, a stage
  or a taxonomy
- [Stages](docs/README.md#stages) and [internals](docs/README.md#internals):
  one page per stage and per layer

## Published datasets

Datasets REDACT has produced, on Hugging Face under
[centrepourlasecuriteia](https://huggingface.co/centrepourlasecuriteia).
Access is granted on request, on each dataset's page.

- [content-moderation-input-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/content-moderation-input-dataset)
- [content-moderation-output-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/content-moderation-output-dataset)
- [jailbreak-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/jailbreak-dataset)
- [BET-jailbreak-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/BET-jailbreak-dataset)
- [constitution-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/constitution-dataset)
- [constitution-input-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/constitution-input-dataset)
- [constitution-input-augmented-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/constitution-input-augmented-dataset)

## License

MIT License. See [LICENSE](LICENSE).
