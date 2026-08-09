# ai4s-myGPT

A compact, readable implementation of a decoder-only GPT language model for the
AI4S final project. The repository is deliberately small enough to study: it
contains a character tokenizer, causal self-attention, a training loop,
checkpointing, evaluation, and text generation.

## Project layout

```text
configs/              Model and training presets
src/mygpt/            Reusable Python package
scripts/              Command-line entry points
tests/                Unit and smoke tests
notebooks/            Optional experiments
outputs/              Generated data/checkpoints (gitignored)
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Apple Silicon, PyTorch will use MPS automatically when it is available.

## Quick start

The default presets train on a local text export of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Prepare it
once before training. The preparation command streams 10,000 stories by default,
which keeps this character-level teaching implementation practical.

```bash
# Download and export the default TinyStories subset
python scripts/prepare_data.py

# A short smoke-training run
python scripts/train.py --config configs/gpt_small.yaml --max-steps 20

# Inspect validation loss from the final checkpoint
python scripts/evaluate.py --checkpoint outputs/gpt-small/last.pt

# Generate text
python scripts/sample.py --checkpoint outputs/gpt-small/last.pt \
  --prompt "The " --max-new-tokens 120
```

To use your own UTF-8 corpus instead:

```bash
python scripts/prepare_data.py --input path/to/corpus.txt \
  --output outputs/data/custom.txt
python scripts/train.py --config configs/gpt_small.yaml \
  --data outputs/data/custom.txt
```

Use `--max-stories` to change the downloaded TinyStories subset size. The
upstream dataset has separate train and validation splits; this compact pipeline
streams the requested training stories and applies its configured contiguous
train/validation split locally.

Run the tests with:

```bash
pytest -q
```

## Configuring an experiment

Every experiment is controlled by a YAML file. Important settings are:

- `model.block_size`: maximum context length.
- `model.n_layer`, `n_head`, `n_embd`: Transformer size.
- `data.path`: UTF-8 text file; both bundled presets default to the prepared
  `outputs/data/tinystories.txt` corpus.
- `training.max_steps`: number of optimizer updates.
- `training.device`: `auto`, `cpu`, `mps`, or `cuda`.

`gpt_small.yaml` is intended for learning and smoke tests. The medium preset is
still educational rather than production-scale, but requires substantially more
memory and training time.

## Notes

- Checkpoints include the model, optimizer, configuration, tokenizer vocabulary,
  step, and best validation loss.
- The validation split is the final contiguous portion of the text. For serious
  experiments, provide a sufficiently large corpus and treat this split as a
  development set rather than repeatedly tuning on a final test set.
- Generated prose from a short demo run is only a pipeline check; coherent text
  requires more data and training.
